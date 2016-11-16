//
//  LBF.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
#include "Config.h"

using namespace std;
using namespace cv;

// parameters
Config global_params;


string modelPath ="./../";
string dataPath = "./../../face-detection-dataset-300W/";
string cascadeName = "haarcascade_frontalface_alt.xml";
