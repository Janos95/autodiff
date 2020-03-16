#include "autodiff.hpp"

#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

using DiffVec = Eigen::Matrix<DiffVar<double>, Eigen::Dynamic, 1>;
using DiffMatrix = Eigen::Matrix<DiffVar<double>, Eigen::Dynamic, Eigen::Dynamic>;
using DiffLowerTriangView = Eigen::TriangularView<DiffMatrix, Eigen::Lower>;

class LogLikelihood
{
public:
    LogLikelihood(DiffMatrix& C, DiffVec& M): Cov(C), Mu(M), lu(C)
    {
    }

    DiffMatrix& Cov;
    DiffVec& Mu;
    Eigen::FullPivLU<DiffMatrix> lu;

    auto operator()(Eigen::VectorXd const& X)
    {
        return -.5 * (log(abs(lu.determinant())) + (X-Mu).transpose() * lu.inverse() * (X-Mu));
    }
};


auto randomNormalDistrib(int n)
{
    Eigen::VectorXd Mu = Eigen::VectorXd::Random(n);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n,n);
    Eigen::MatrixXd Cov = .5 * (A + A.transpose()) + n * Eigen::MatrixXd::Identity(n, n);
    return std::make_pair(Cov, Mu);
}

template<class DerivedC, class DerivedM>
auto draw(Eigen::MatrixBase<DerivedC>& Cov, Eigen::MatrixBase<DerivedM>& Mu, int numSamples){
    Eigen::LLT<DerivedC> llt(Cov);
    DerivedC L = llt.matrixL();
    std::vector<DerivedM> Xs(numSamples);
    std::default_random_engine engine(0);
    std::normal_distribution<typename DerivedM::Scalar> distr;
    auto n = Mu.size();
    for(auto& X : Xs){
        DerivedM z(n);
        for (int i = 0; i < n; ++i) {
            z[i] = distr(engine);
        }
        X = L*z + Mu;
    }
    return Xs;
}

struct Cost
{
    std::vector<DiffVec> Xs;

    auto operator(auto Cov, auto Mu)(){
        LogLikelihood llh(Cov, Mu);
        return std::accumulate(Xs.begin(), Xs.end(), 0, llh);
    }
};

struct Constraint
{
    auto operator(auto L)(){
        return L.diagonal();
    }
};


int main() {
    int dim = 100;
    int numSamples = 1000;
    auto [Cov, Mu] = randomNormalDistrib(dim);

    Cost cost{draw(Cov, Mu, numSamples)};
    Constraint constraint();

    DiffVec MuDiff = DiffVec::Zero(dim);
    DiffVec LDiffData(dim * (dim + 1) / 2);
    Eigen::Map<DiffLowerTriangView> LDiff(LDiffData.data(), dim, dim);

    Problem problem;
    problem.addCost(cost, CovDiff, MuDiff);
    problem.addConstraint(constraint, lower, upper);
    Solver(problem, options, summary);
}
