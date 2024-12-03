#ifndef TEST_HUNGARIAN_HH_
#define TEST_HUNGARIAN_HH_
/**
 * Solution to https://open.kattis.com/problems/cordonbleu using Hungarian
 * algorithm.
 */

#include <limits>
#include <vector>
#include <algorithm>


template <class DT>
std::vector<int> new_hungarian_algorithm(const std::vector<std::vector<DT>> &graph) {
  const int n = graph.size();
  std::vector<int> match(n, -1);
  std::vector<DT> u(n, 0);
  std::vector<DT> v(n, 0);
  for (int k = 0; k < n; ++k) {
    std::vector<int> previous(n, -1);
    std::vector<bool> scanned(n, false);
    std::vector<DT> dist(n, std::numeric_limits<DT>::max());
    int i = k;
    int selected_col;

    while (i != -1) {
      DT dmin = std::numeric_limits<DT>::max();
      scanned.at(i) = true;
      bool found = false;

      for (int j = 0; j < n; ++j) {
        if (graph.at(i).at(j) - u.at(i) - v.at(j) < dist.at(j)) {
          dist.at(j) = graph.at(i).at(j) - u.at(i) - v.at(j);
          previous.at(j) = i;
          if (dist.at(j) == 0) {
            selected_col = j;
            found = true;
          }
        }
      }
      if (!found) {
        for (int l = 0; l < n; ++l) {
          if (dist.at(l) > 0 && dist.at(l) < dmin) {
            dmin = dist.at(l);
            if (match.at(l) == -1 || !scanned.at(match.at(l))) {
              selected_col = l;
            }
          }
        }
        // Update
        for (int row = 0; row < k; ++row) {
          if (scanned.at(row)) {
            u.at(row) += dmin;
          }
        }
        for (int col = 0; col < n; ++col) {
          if (std::abs(dist.at(col)) < 1e-12) {
            v.at(col) -= dmin;
          } else {
            dist.at(col) -= dmin;
          }
        }
      }
      for (int col = 0; col < n; ++col) {
        if (std::abs(dist.at(col)) < 1e-12) {
          if (match.at(col) == -1) {
            selected_col = col;
            i = match.at(col);
            break;
          } else if (!scanned.at(match.at(col))) {
            i = match.at(col);
          }
        }
      }
    }


    while (selected_col != n) {
      int row = previous.at(selected_col);
      int next_col = std::find(match.begin(), match.end(), row) - match.begin();
      match.at(selected_col) = row;
      selected_col = next_col;
    }
  }
  return match;
}

#endif
