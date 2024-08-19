/**
 * Solution to https://open.kattis.com/problems/cordonbleu using Hungarian
 * algorithm.
 */

#include <cassert>
#include <limits>
#include <vector>

#include<iostream>

/**
 * Sets a = min(a, b)
 * @return true if b < a
 */
template <class T> bool ckmin(T &a, const T &b) { return b < a ? a = b, 1 : 0; }

// TODO(Georg): How to Cite: https://en.wikipedia.org/w/index.php?title=Hungarian_algorithm&oldid=1235721596
/** * Given J jobs and W workers (J <= W), computes the minimum cost to assign each
 * prefix of jobs to distinct workers.
 *
 * @tparam T a type large enough to represent integers on the order of J *
 * max(|C|)
 * @param C a matrix of dimensions JxW such that C[j][w] = cost to assign j-th
 * job to w-th worker (possibly negative)
 *
 * @return a vector of length J, with the j-th entry equaling the minimum cost
 * to assign the first (j+1) jobs to distinct workers
 */
//template <class T> std::vector<T> hungarian(const std::vector<std::vector<T>> &C) {
template <class T> std::vector<int> hungarian(const std::vector<std::vector<T>> &C) {
    const int J = (int)size(C), W = (int)size(C[0]);
    assert(J <= W);
    // job[w] = job assigned to w-th worker, or -1 if no job assigned
    // note: a W-th worker was added for convenience
    std::vector<int> job(W + 1, -1);
    std::vector<T> ys(J), yt(W + 1);  // potentials
    // -yt[W] will equal the sum of all deltas
    std::vector<T> answers;
    std::vector<int> order;
    const T inf = std::numeric_limits<T>::max();
    for (int j_cur = 0; j_cur < J; ++j_cur) {  // assign j_cur-th job
        int w_cur = W;
        job[w_cur] = j_cur;
        // min reduced cost over edges from Z to worker w
        std::vector<T> min_to(W + 1, inf);
        std::vector<int> prv(W + 1, -1);  // previous worker on alternating path
        std::vector<bool> in_Z(W + 1);    // whether worker is in Z
        while (job[w_cur] != -1) {   // runs at most j_cur + 1 times
            in_Z[w_cur] = true;
            const int j = job[w_cur];
            T delta = inf;
            int w_next;
            for (int w = 0; w < W; ++w) {
                if (!in_Z[w]) {
                    if (ckmin(min_to[w], C[j][w] - ys[j] - yt[w]))
                        prv[w] = w_cur;
                    if (ckmin(delta, min_to[w])) w_next = w;
                }
            }
            // delta will always be nonnegative,
            // except possibly during the first time this loop runs
            // if any entries of C[j_cur] are negative
            for (int w = 0; w <= W; ++w) {
                if (in_Z[w]) ys[job[w]] += delta, yt[w] -= delta;
                else min_to[w] -= delta;
            }
            w_cur = w_next;
        }
        // update assignments along alternating path
        for (int w; w_cur != W; w_cur = w) job[w_cur] = job[w = prv[w_cur]];
        answers.push_back(-yt[W]);
    }
    //return answers;
    return job;
}

/**
 * Sanity check: https://en.wikipedia.org/wiki/Hungarian_algorithm#Example
 * First job (5):
 *   clean bathroom: Bob -> 5
 * First + second jobs (9):
 *   clean bathroom: Bob -> 5
 *   sweep floors: Alice -> 4
 * First + second + third jobs (15):
 *   clean bathroom: Alice -> 8
 *   sweep floors: Carol -> 4
 *   wash windows: Bob -> 3
 */
void sanity_check_hungarian() {
  std::vector<std::vector<int>> costs{{8, 5, 9}, {4, 2, 4}, {7, 3, 8}};
    assert((hungarian(costs) == std::vector<int>{5, 9, 15}));
    std::cerr << "Sanity check passed.\n";
}

