#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// UTF-8 strings are tokens here; never split a Chinese character into bytes.
const std::vector<std::string> vocab = {u8"小", u8"明", u8"喜", u8"欢", u8"猫", u8"。", "<EOS>"};
constexpr int V = 7, D = 4, F = 8, MAX_T = 8, EOS = 6;
using Matrix = std::vector<std::vector<double>>;
Matrix zeros(int r, int c) { return Matrix(r, std::vector<double>(c, 0.0)); }

struct Model {
    // All trainable arrays are stored in one flat vector.
    std::vector<double> w;
    int emb, pos, wq, wk, wv, wo, w1, b1, w2, b2, head, bias;
    int alloc(int n) //给一个double数组增加n个位置，返回原先没有增加前的个数
    { 
        int start = static_cast<int>(w.size());
        w.resize(start + n); 
        return start; 
    }
    Model() 
    {
        emb = alloc(V*D); 
        pos = alloc(MAX_T*D);
        wq = alloc(D*D); 
        wk = alloc(D*D); 
        wv = alloc(D*D); 
        wo = alloc(D*D);
        w1 = alloc(D*F); 
        b1 = alloc(F); 
        w2 = alloc(F*D); 
        b2 = alloc(D);
        head = alloc(D*V); 
        bias = alloc(V);
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(0.0, 0.25);
        for (double& value : w) value = normal(rng);
    }

    // X[T,in] * W[in,out] -> Y[T,out]
    Matrix linear(const Matrix& x, int offset, int out, int b = -1) const 
    {
        int t = static_cast<int>(x.size()), in = static_cast<int>(x[0].size());
        Matrix y = zeros(t, out);
        for (int i=0; i<t; ++i)
            for (int j=0; j<out; ++j) 
            {
                y[i][j] = b < 0 ? 0.0 : w[b+j];
                for (int k=0; k<in; ++k) 
                    y[i][j] += x[i][k]*w[offset+k*out+j];
            }
        return y;
    }
    // Simplified RMSNorm: no learned scale.
    Matrix norm(Matrix x) const {
        for (auto& row : x) {
            double sum=0;
            for (double value : row) sum += value*value;
            double divisor = std::sqrt(sum / row.size() + 1e-5);
            for (double& value : row) value /= divisor;
        }
        return x;
    }

    Matrix forward(const std::vector<int>& ids) const {
        int t = static_cast<int>(ids.size());
        if (t == 0 || t > MAX_T) throw std::runtime_error("Invalid sequence length");
        Matrix x = zeros(t,D);
        for (int i=0; i<t; ++i)
            for (int k=0; k<D; ++k) x[i][k] = w[emb+ids[i]*D+k]+w[pos+i*D+k];

        Matrix n = norm(x);
        Matrix q = linear(n,wq,D), key = linear(n,wk,D), value = linear(n,wv,D);
        Matrix a = zeros(t,D);
        for (int i=0; i<t; ++i) {
            std::vector<double> scores(i+1);
            // Causal mask: position i can only attend to positions 0..i.
            for (int j=0; j<=i; ++j) {
                double dot=0;
                for (int k=0; k<D; ++k) dot += q[i][k]*key[j][k];
                scores[j] = dot/std::sqrt(double(D));
            }
            double maximum = *std::max_element(scores.begin(),scores.end());
            double sum=0;
            for (double& s : scores) { s=std::exp(s-maximum); sum+=s; }
            for (int j=0; j<=i; ++j)
                for (int k=0; k<D; ++k) a[i][k] += scores[j]/sum*value[j][k];
        }
        Matrix projected = linear(a,wo,D);
        for (int i=0; i<t; ++i)
            for (int k=0; k<D; ++k) x[i][k] += projected[i][k];

        Matrix f = linear(norm(x),w1,F,b1);
        for (auto& row : f) for (double& s : row) s=std::tanh(s);
        Matrix f_out = linear(f,w2,D,b2);
        for (int i=0; i<t; ++i)
            for (int k=0; k<D; ++k) x[i][k] += f_out[i][k];
        return linear(norm(x),head,V,bias);
    }

    double loss(const std::vector<int>& x, const std::vector<int>& target) const 
    {
        Matrix logits = forward(x);
        double total=0;
        for (int i=0; i<static_cast<int>(x.size()); ++i) {
            double m=*std::max_element(logits[i].begin(),logits[i].end()), sum=0;
            for (double s : logits[i]) sum+=std::exp(s-m);
            total += m+std::log(sum)-logits[i][target[i]];
        }
        return total/x.size();
    }
    void save(const std::string& path) const 
    {
        std::ofstream file(path);
        if (!file) throw std::runtime_error("Cannot write weights");
        file << "TINY_TRANSFORMER_V1 " << D << ' ' << F << ' ' << MAX_T << ' ' << w.size() << '\n';
        file << std::setprecision(17);
        for (double value : w) file << value << '\n';
        if (!file) throw std::runtime_error("Failed to save weights");
    }
    void load(const std::string& path) {
        std::ifstream file(path);
        std::string magic; int d,f,t; size_t count;
        if (!(file>>magic>>d>>f>>t>>count) || magic!="TINY_TRANSFORMER_V1" || d!=D || f!=F || t!=MAX_T || count!=w.size())
            throw std::runtime_error("Missing or incompatible weights; run train first");
        for (double& value : w) if (!(file>>value)) throw std::runtime_error("Truncated weights");
    }
};

void train(Model& model, const std::string& path) 
{
    const std::vector<int> x={0,1,2,3,4,5};
    const std::vector<int> target={1,2,3,4,5,EOS};
    //fprintf(stdout,"model.w.size() = %ld\n",model.w.size());
    std::vector<double> grad(model.w.size()); //梯度235长度
    std::vector<double> m(model.w.size(),0);
    std::vector<double> v(model.w.size(),0);
    constexpr double eps=1e-4; 
    constexpr double lr=0.03;
    std::cout << "Trainable parameters: " << model.w.size() << '\n';
    for (int step=1; step<=400; ++step) {
        // Central finite differences: explicit gradient, no autograd library.
        // All gradients are computed at the SAME weights before the update.
        for (size_t p=0; p<model.w.size(); ++p)//检查模型权重，计算每个权重的梯度
        {
            double original=model.w[p];//保存当前权重
            model.w[p]=original+eps; //增大权重
            double plus=model.loss(x,target);//计算损失
            model.w[p]=original-eps; //减少权重
            double minus=model.loss(x,target);//计算损失
            model.w[p]=original;//恢复原来权重
            grad[p]=(plus-minus)/(2*eps);
        }
        for (size_t p=0; p<model.w.size(); ++p) {
            m[p]=0.9*m[p]+0.1*grad[p];
            v[p]=0.999*v[p]+0.001*grad[p]*grad[p];
            double mh=m[p]/(1-std::pow(0.9,step));
            double vh=v[p]/(1-std::pow(0.999,step));
            model.w[p]-=lr*mh/(std::sqrt(vh)+1e-8);
        }
        if (step==1 || step%50==0) std::cout << "step=" << step << " loss=" << model.loss(x,target) << '\n';
    }
    model.save(path);
    std::cout << "Saved: " << path << '\n';
}

std::vector<int> tokenize(const std::string& text) {
    std::vector<int> ids;
    for (size_t p=0; p<text.size();) {
        bool found=false;
        for (int i=0; i<EOS; ++i) if (text.compare(p,vocab[i].size(),vocab[i])==0) {
            ids.push_back(i); p+=vocab[i].size(); found=true; break;
        }
        if (!found) throw std::runtime_error("Prompt contains a token outside the tiny vocabulary");
    }
    return ids;
}
void infer(Model& model, const std::string& prompt) {
    std::vector<int> ids=tokenize(prompt);
    std::cout << "Input: " << prompt << "\nOutput: " << prompt;
    bool ended=false;
    while (ids.size()<MAX_T) {
        // Recompute full context for clarity. No KV cache in this example.
        Matrix logits=model.forward(ids);
        const auto& last=logits.back();
        int next=static_cast<int>(std::max_element(last.begin(),last.end())-last.begin());
        if (next==EOS) { ended=true; break; }
        std::cout << vocab[next]; ids.push_back(next);
    }
    std::cout << (ended ? "\n" : " [context limit]\n");
}
int main(int argc,char** argv) {
    try {
        Model model;
        std::string mode=argc>1?argv[1]:"help";
        std::string path=argc>2?argv[2]:"weights.txt";
        if (mode=="train") train(model,path);
        else if (mode=="infer") { model.load(path); infer(model,argc>3?argv[3]:u8"小明"); }
        else { std::cout << "Usage: tiny_transformer train [weights.txt]\n"
                              "       tiny_transformer infer [weights.txt] [UTF-8 prompt]\n"; return 1; }
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}


