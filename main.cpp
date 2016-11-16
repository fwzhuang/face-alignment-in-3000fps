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

vector<Rect> detect_faces(Mat & small_img, CascadeClassifier& cascade)
{
    vector<Rect> faces;

    double t = (double)cvGetTickCount();
    cascade.detectMultiScale( small_img, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );

    if( global.config.try_flip ){
        vector<Rect> fliped_faces;
        flip(small_img, small_img, 1);
        cascade.detectMultiScale( small_img, fliped_faces,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for(Rect &r : fliped_faces)
        {
            faces.push_back(Rect(small_img.cols - r.x - r.width, r.y, r.width, r.height));
        }
    }

    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    return faces;
}

vector<Rect> detect_faces_with_scale(Mat & img, CascadeClassifier& cascade, double scale)
{
    Mat small_img( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    resize( img, small_img, small_img.size(), 0, 0, INTER_LINEAR );
    equalizeHist( small_img, small_img );

    vector<Rect> faces = detect_faces(small_img, cascade);

    for (Rect & r : faces) {
        r.x *= scale;
        r.y *= scale;
        r.width *= scale;
        r.height *= scale;
    }

    return faces;
}

Mat_<double> align_face(Mat & gray, Rect & r, LBFRegressor& regressor)
{
    double t = (double)cvGetTickCount();
    Point center;
    BoundingBox boundingbox;

    boundingbox.start_x = r.x;
    boundingbox.start_y = r.y;
    boundingbox.width   = r.width - 1;
    boundingbox.height  = r.height - 1;
    boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
    boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;

    Mat_<double> current_shape = regressor.Predict(gray, boundingbox);

    t = (double)cvGetTickCount() - t;
    printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    return current_shape;
}

vector<Mat_<double>> align_faces(Mat & gray, vector<Rect> & faces, LBFRegressor& regressor)
{
    vector<Mat_<double>> result;
    for(Rect& r : faces) {
        result.push_back(align_face(gray, r, regressor));
    }
    return result;
}



void draw_faces(Mat & img, vector<Mat_<double>> &shapes)
{
    int save_count=0;
    for(auto current_shape: shapes) {
        // draw bounding box

        // rectangle(out_img, cvPoint(boundingbox.start_x,boundingbox.start_y),
        //           cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);

        // draw result :: red
        for(int i = 0; i < global.config.landmark_num; i++){
            circle(img, Point2d(current_shape(i,0), current_shape(i,1)), 3, Scalar(255,255,255), -1, 8, 0);
        }
    }

    cv::imshow( "result", img );
    char a = waitKey(0);
    if (a=='s') {
        save_count++;
        imwrite(to_string(save_count)+".jpg", img);
    }
}

void align_image_and_draw(Mat& img,
                          CascadeClassifier& cascade,
                          LBFRegressor& regressor)
{
    Mat gray;
    cvtColor( img, gray, CV_BGR2GRAY );

    vector<Rect> faces = detect_faces_with_scale(gray, cascade, global.config.scale);
    vector<Mat_<double>> shapes = align_faces(gray, faces, regressor);
    draw_faces(img, shapes);
}

void align_captures_and_draw(CvCapture *capture,
                             CascadeClassifier& cascade,
                             LBFRegressor& regressor)
{
    Mat frame, frameCopy;

    for(;;) {
        IplImage* iplImg = cvQueryFrame(capture);
        frame = cvarrToMat(iplImg);
        if( frame.empty() )
            break;
        if( iplImg->origin == IPL_ORIGIN_TL )
            frame.copyTo( frameCopy );
        else
            flip( frame, frameCopy, 0 );

        align_image_and_draw( frameCopy, cascade,regressor);

        if( waitKey( 10 ) >= 0 )
            break;
    }

    cvReleaseCapture( &capture );
}

void align_filename_list_and_draw(string filename,
                                  CascadeClassifier& cascade,
                                  LBFRegressor& regressor)
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
                align_image_and_draw(image, cascade, regressor);
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

int align_inputname(string input_name) {
    CvCapture* capture = 0;
    CascadeClassifier cascade;
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
    // -- 0. Load LBF model
    LBFRegressor regressor;
    regressor.Load(global.config.model_path + "LBF.model");

    // -- 1. Load the cascades
    if( !cascade.load( global.config.cascade_name ) ){
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    // cvNamedWindow( "result", 1 );
    // -- 2. Read the video stream
    if( capture ){
        align_captures_and_draw(capture, cascade, regressor);
    }
    else if( !image.empty() ){
        align_image_and_draw( image, cascade, regressor);
        waitKey(0);
    }
    else if( !input_name.empty() ){
        align_filename_list_and_draw(input_name, cascade, regressor);
    }

    cvDestroyWindow("result");

    return 0;
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
            return align_inputname("");
        }
        else if(argc ==3){
            return align_inputname(argv[2]);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}
