#ifndef App_H
#define App_H

#include <string>
#include <vector>
#include "Config.h"
#include "LBFRegressor.h"

using namespace std;
using namespace cv;

namespace FaceAlignment {

class App {
 public:

  Config config;

  vector<Rect> detect_faces(Mat & small_img);

  vector<Rect> detect_faces_with_scale(Mat & img, double scale);

  Mat_<double> align_face(Mat & gray, Rect & r);

  vector<Mat_<double>> align_faces(Mat & gray, vector<Rect> & faces);

  void draw_faces(Mat & img, vector<Mat_<double>> &shapes);

  void align_image_and_draw(Mat& img);

  void align_captures_and_draw(CvCapture *capture);

  void align_filename_list_and_draw(string filename);




  CascadeClassifier cascade;
  LBFRegressor regressor;


};

}

#endif
