#include "LBF.h"
#include "LBFRegressor.h"
#include "App.h"

using namespace std;
using namespace cv;

namespace FaceAlignment {

void PrintHelp(){
    cout << "Usage:"<<endl;
    cout << "1. train your own model:    LBF.out TrainModel "<<endl;
    cout << "2. test model on dataset:   LBF.out TestModel"<<endl;
    cout << "3. test model via a camera: LBF.out Demo "<<endl;
    cout << "4. test model on a pic:     LBF.out Demo xx.jpg"<<endl;
    cout << "5. test model on pic set:   LBF.out Demo Img_Path.txt"<<endl;
    cout << endl;

}

void align_filename_list_and_draw(App & app, string filename)
{
    /* Input is a text file containing the list of the image filenames to be
       processed - one per line */

    Mat image;

    FILE* f = fopen( filename.c_str(), "rt" );
    if( f ){
        char buf[1000+1];
        while( fgets( buf, 1000, f ) ){
            int len = (int)strlen(buf), c;
            while( len > 0 && isspace(buf[len-1]) )
                len--;
            buf[len] = '\0';
            cout << "file " << buf << endl;
            image = imread( buf, 1 );
            if( !image.empty() ){
                app.align_image_and_draw(image);
                c = waitKey(0);
                if( c == 27 || c == 'q' || c == 'Q' )
                    break;
            }
            else{
                cerr << "Aw snap, couldn't read image " << buf << endl;
            }
        }
        fclose(f);
    }

}


int align_inputname(App & app, string input_name) {
    CvCapture* capture = 0;
    Mat image;

    // name is empty or a number
    if( input_name.empty() || (isdigit(input_name.c_str()[0]) && input_name.c_str()[1] == '\0') ){
        capture = cvCaptureFromCAM( input_name.empty() ? 0 : input_name.c_str()[0] - '0' );
        int c = input_name.empty() ? 0 : input_name.c_str()[0] - '0' ;
        if(!capture){
            cout << "Capture from CAM " <<  c << " didn't work" << endl;
            return -1;
        }
    }
    // name is not empty
    else if( input_name.size() ){
        if (input_name.find(".jpg")!=string::npos||input_name.find(".png")!=string::npos
            ||input_name.find(".bmp")!=string::npos){
            image = imread( input_name, 1 );
            if (image.empty()){
                cout << "Read Image fail" << endl;
                return -1;
            }
        }
        else if(input_name.find(".mp4")!=string::npos||input_name.find(".avi")!=string::npos
                ||input_name.find(".wmv")!=string::npos){
            capture = cvCaptureFromAVI( input_name.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
            return -1;
        }
    }


    // cvNamedWindow( "result", 1 );
    // -- 2. Read the video stream
    if( capture ){
        app.align_captures_and_draw(capture);
    }
    else if( !image.empty() ){
        app.align_image_and_draw(image);
        waitKey(0);
    }
    else if( !input_name.empty() ){
        align_filename_list_and_draw(app, input_name);
    }

    cvDestroyWindow("result");

    return 0;
}

}

using namespace FaceAlignment;

int main( int argc, const char** argv ){
    App app(global_config.cascade_name,
	    global_config.model_path + "LBF.model",
	    global_config.model_path + "Regressor.model");

    bool training = (argc > 1 && strcmp(argv[1], "TrainModel") == 0);

    if (argc == 1){
        PrintHelp();
        exit(0);
    }

    if (! training) {
      global_config.read(global_config.model_path + "LBF.model");
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
            return align_inputname(app, "");
        }
        else if(argc ==3){
            return align_inputname(app, argv[2]);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}
