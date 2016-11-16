#include "App.h"

namespace FaceAlignment {

vector<Rect> App::detect_faces(Mat & small_img)
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

    if( global_config.try_flip ){
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

vector<Rect> App::detect_faces_with_scale(Mat & img, double scale)
{
    Mat small_img( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    resize( img, small_img, small_img.size(), 0, 0, INTER_LINEAR );
    equalizeHist( small_img, small_img );

    vector<Rect> faces = detect_faces(small_img);

    for (Rect & r : faces) {
        r.x *= scale;
        r.y *= scale;
        r.width *= scale;
        r.height *= scale;
    }

    return faces;
}

Mat_<double> App::align_face(Mat & gray, Rect & r)
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

vector<Mat_<double>> App::align_faces(Mat & gray, vector<Rect> & faces)
{
    vector<Mat_<double>> result;
    for(Rect& r : faces) {
        result.push_back(align_face(gray, r));
    }
    return result;
}



void App::draw_faces(Mat & img, vector<Mat_<double>> &shapes)
{
    int save_count=0;
    for(auto current_shape: shapes) {
        // draw bounding box

        // rectangle(out_img, cvPoint(boundingbox.start_x,boundingbox.start_y),
        //           cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);

        // draw result :: red
        for(int i = 0; i < global_config.landmark_num; i++){
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

void App::align_image_and_draw(Mat& img)
{
    Mat gray;
    cvtColor( img, gray, CV_BGR2GRAY );

    vector<Rect> faces = detect_faces_with_scale(gray, global_config.scale);
    vector<Mat_<double>> shapes = align_faces(gray, faces);
    draw_faces(img, shapes);
}

void App::align_captures_and_draw(CvCapture *capture)
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

        align_image_and_draw( frameCopy);

        if( waitKey( 10 ) >= 0 )
            break;
    }

    cvReleaseCapture( &capture );
}

}