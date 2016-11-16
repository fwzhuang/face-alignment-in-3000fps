#include "Config.h"
#include <iostream>
#include <fstream>

using namespace std;

Config::Config() {
  bagging_overlap = 0.4;
  max_numtrees = 10;
  max_depth = 5;
  max_numthreshs = 500;
  landmark_num = 68;
  initial_num = 5;

  max_numstage = 7;
  double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05};
  double m_max_numfeats[10] = {500, 500, 500, 300, 300, 200, 200,200,100,100};
  for (int i=0;i<10;i++){
    max_radio_radius[i] = m_max_radio_radius[i];
  }
  for (int i=0;i<10;i++){
    max_numfeats[i] = m_max_numfeats[i];
  }
}

void Config::read(string path) {
  ifstream file;
  file.open(path);
  file >> bagging_overlap;
  file >> max_numtrees;
  file >> max_depth;
  file >> max_numthreshs;
  file >> landmark_num;
  file >> initial_num;
  file >> max_numstage;

  for (int i = 0; i< max_numstage; i++){
    file >> max_radio_radius[i];
  }

  for (int i = 0; i < max_numstage; i++){
    file >> max_numfeats[i];
  }
  file.close();
}
