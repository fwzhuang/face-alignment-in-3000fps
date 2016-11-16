#ifndef Config_H
#define Config_H
#include <string>

using namespace std;

namespace FaceAlignment {

class Config {

 public:

  Config();
  void read(std::string path);
  void read(std::ifstream &file);
  void write(std::ofstream &file);

  double bagging_overlap;
  int max_numtrees;
  int max_depth;
  int landmark_num;// to be decided
  int initial_num;

  int max_numstage;
  double max_radio_radius[10];
  int max_numfeats[10]; // number of pixel pairs
  int max_numthreshs;

  string model_path;
  string data_path;
  string cascade_name;

  const double scale;
  const bool try_flip;
};
}

#endif
