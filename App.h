#ifndef App_H
#define App_H

#include <string>
#include <vector>
#include "Config.h"
#include "LBFRegressor.h"
#include <boost/python.hpp>

using namespace std;
using namespace cv;
namespace py = boost::python;

namespace FaceAlignment {

class App {
 public:

  App();

  App(string cascade_model_path,
      string lbf_model_path,
      string regressor_model_path);

  vector<Rect> detect_faces(Mat & small_img);

  vector<Rect> detect_faces_with_scale(Mat & img, double scale);

  Mat_<double> align_face(Mat & gray, Rect & r);

  py::object align_face_with_ptr(int rows, int cols, long img_data, Rect & r);

  vector<Mat_<double>> align_faces(Mat & gray, vector<Rect> & faces);

  void draw_face(Mat & img, Mat_<double> &shape);

  void show_faces(Mat & img, vector<Mat_<double>> &shapes);

  void align_image_and_show(Mat& img);

  void align_captures_and_show(CvCapture *capture);

  void align_filename_list_and_show(string filename);

private:

  CascadeClassifier cascade;
  LBFRegressor regressor;


};

}

#endif
