/**
 * Solution to https://open.kattis.com/problems/cordonbleu using Hungarian
 * algorithm.
 */

#include <limits>
#include <numeric>
#include <vector>

#include<iostream>

template <class DT>
std::vector<int> new_hungarian_algorithm(const std::vector<std::vector<DT>> &graph) {
  const int n = graph.size();
  std::vector<int> match(n+1, -1);
  std::vector<DT> ya(n, 0);
  std::vector<DT> yb(n, 0);
  for (int i = 0; i < n; ++i) {
    int cur_b = n;
    match.at(n) = i;
    std::vector<DT> dist(n, std::numeric_limits<DT>::max());
    std::vector<int> previous(n, -1);
    std::vector<bool> Z(n + 1, false);
    while (match.at(cur_b) != -1) {
      Z.at(cur_b) = true;
      const int a = match.at(cur_b);
      DT delta = std::numeric_limits<DT>::max();
      int next_b = -1;

      for (int b = 0; b < n; ++b) {
        if(!Z.at(b)) {
          if (graph.at(a).at(b) - ya.at(a) - yb.at(b) < dist.at(b)) {
            dist.at(b) = graph.at(a).at(b) - ya.at(a) - yb.at(b);
            previous.at(b) = cur_b;
          }
          if(dist.at(b) < delta) {
            delta = dist.at(b);
            next_b = b;
          }
        }
      }
      for (int b = 0; b < n; ++b) {
        if (Z.at(b)) {
          yb.at(b) -= delta;
          ya.at(match.at(b)) += delta;
        } else {
          dist.at(b) -= delta;
        }
      }
      cur_b = next_b;
    }
    for (;cur_b != n;) {
      match.at(cur_b) = match.at(previous.at(cur_b));
      cur_b = previous.at(cur_b);
    }
  }
  match.erase(match.end());
  return match;
}
