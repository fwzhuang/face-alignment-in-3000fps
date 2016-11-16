#include "LBF.h"
#include "LBFRegressor.h"

using namespace std;
using namespace cv;
using namespace FaceAlignment;

void PrintHelp(){
    cout << "Usage:"<<endl;
    cout << "1. train your own model:    LBF.out TrainModel "<<endl;
    cout << "2. test model on dataset:   LBF.out TestModel"<<endl;
    cout << "3. test model via a camera: LBF.out Demo "<<endl;
    cout << "4. test model on a pic:     LBF.out Demo xx.jpg"<<endl;
    cout << "5. test model on pic set:   LBF.out Demo Img_Path.txt"<<endl;
    cout << endl;

}


int main( int argc, const char** argv ){

    bool training = (argc > 1 && strcmp(argv[1], "TrainModel") == 0);

    if (argc == 1){
        PrintHelp();
        exit(0);
    }

    if (! training) {
      global.config.read(global.config.model_path + "LBF.model");
    }

    if(strcmp(argv[1],"TrainModel")==0){
        vector<string> train_data_name = {"afw", "helen", "lfpw"};
        TrainModel(train_data_name);
    }
    else if (strcmp(argv[1], "TestModel")==0){
        vector<string> testDataName = {"ibug"};
        double MRSE = TestModel(testDataName);
    }
    else if (strcmp(argv[1], "Demo")==0){
        if (argc == 2){
            return FaceDetectionAndAlignment("");
        }
        else if(argc ==3){
            return FaceDetectionAndAlignment(argv[2]);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}
