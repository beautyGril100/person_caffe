
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>

#include <string>
#include <vector>
#include <sys/time.h>
#include <glog/logging.h>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include "image.h"
#include "yolo_layer.h"

using namespace caffe;
using namespace cv;

#if 0
//#define CLASS_NUM		81	// 80 classes + background class ==> 81 classes
const char *class_label[] = {
	"_background_",
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" 
};


bool signal_recieved = false;

void sig_handler(int signo){
    if( signo == SIGINT ){
            printf("received SIGINT\n");
            signal_recieved = true;
    }
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

int main( int argc, char** argv )
{
    string model_file;
    string weights_file;
    string image_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_path]";
        return -1;
    }	

    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    image im,sized;
    vector<Blob<float>*> blobs;
    blobs.clear();

    int nboxes = 0;
    int size;
    detection *dets = NULL;
        
    /* load and init network. */
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);
    LOG(INFO) << "net inputs numbers is " << net->num_inputs();
    LOG(INFO) << "net outputs numbers is " << net->num_outputs();

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *net_input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "input data layer channels is  " << net_input_data_blobs->channels();
    LOG(INFO) << "input data layer width is  " << net_input_data_blobs->width();
    LOG(INFO) << "input data layer height is  " << net_input_data_blobs->height();

    size = net_input_data_blobs->channels()*net_input_data_blobs->width()*net_input_data_blobs->height();


    uint64_t beginDataTime =  current_timestamp();
    //load image
    im = load_image_color((char*)image_path.c_str(),0,0);
    printf("func:%s,line:%d, im.h:%d,im.w:%d,im.c:%d\n", __FUNCTION__, __LINE__, im.h, im.w, im.c);
    printf("func:%s,line:%d, network_im.h:%d,network_im.w:%d,network_im.c:%d, network_im.n:%d\n", __FUNCTION__, __LINE__, net_input_data_blobs->height(), net_input_data_blobs->width(), net_input_data_blobs->channels(), net_input_data_blobs->num());
    sized = letterbox_image(im,net_input_data_blobs->width(),net_input_data_blobs->height());
    printf("func:%s,line:%d, sized_im.h:%d,sized_im.w:%d,sized_im.c:%d\n", __FUNCTION__, __LINE__, sized.h, sized.w, sized.c);
    #if 0
    cuda_push_array(net_input_data_blobs->mutable_gpu_data(),sized.data,size);
    #else
    cuda_push_array(net_input_data_blobs->mutable_cpu_data(),sized.data,size);
    #endif

    uint64_t endDataTime =  current_timestamp();
    LOG(INFO) << "processing data operation avergae time is "
              << endDataTime - beginDataTime << " ms";

    uint64_t startDetectTime = current_timestamp();
    // forward
    net->Forward();
    for(int i =0;i<net->num_outputs();++i){
        blobs.push_back(net->output_blobs()[i]);
    }
    dets = get_detections(blobs,im.w,im.h,
        net_input_data_blobs->width(),net_input_data_blobs->height(),&nboxes);

    uint64_t endDetectTime = current_timestamp();
    LOG(INFO) << "caffe yolov3 : processing network yolov3 tiny avergae time is "
              << endDetectTime - startDetectTime << " ms";

    //show detection results
    Mat img = imread(image_path.c_str());
    printf("func:%s,line:%d,img.h:%d,img.w:%d,img.c:%d\r\n", __FUNCTION__, __LINE__, img.rows, img.cols, img.channels());
    int i,j;
    for(i=0;i< nboxes;++i){
        char labelstr[4096] = {0};
        int cls = -1;
        for(j=0;j<80;++j){
            if(dets[i].prob[j] > 0.5){
                if(cls < 0){
                    cls = j;
                }
                LOG(INFO) << "label = " << cls
                          << ", prob = " << dets[i].prob[j]*100;
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
//            LOG(INFO) << "  x = " << b.x
//                      << ", y = " << b.y
//                      << ", w = " << b.w
//                      << ", h = " << b.h;
#if 0
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
#else
            int left  = round((b.x-b.w/2.)*im.w);
            int right = round((b.x+b.w/2.)*im.w);
            int top   = round((b.y-b.h/2.)*im.h);
            int bot   = round((b.y+b.h/2.)*im.h);
#endif
            printf("prob:%f\n", dets[i].prob[cls]);
            char tempText[256];
            sprintf(tempText, "%s(%f)", class_label[cls + 1], dets[i].prob[cls]);

            rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
            putText(img, tempText, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.8, 8);
            LOG(INFO) << "  left = " << left
                      << ", right = " << right
                      << ", top = " << top
                      << ", bot = " << bot;

        }
    }

    namedWindow("show",CV_WINDOW_AUTOSIZE);
    imshow("show",img);
    waitKey(0);

    free_detections(dets,nboxes);
    free_image(im);
    free_image(sized);
        
    LOG(INFO) << "done.";
    return 0;
}
#endif
#if 1
#include "retinaface_anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "retinaface_config.h"
#include "retinaface_tools.h"
#endif
int main(int argc, char** argv ) {
#if 0
    extern float pixel_mean[3];
    extern float pixel_std[3];
	std::string param_path =  "./models/retina.param";
	std::string bin_path = "./models/retina.bin";

	ncnn::Net _net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());
 
#else
    string model_file;
    string weights_file;
    string image_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_path]";
        return -1;
    }	
 // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    //image im,sized;
    vector<Blob<float>*> blobs;
    blobs.clear();

    int nboxes = 0;
    int size;
    detection *dets = NULL;
        
    /* load and init network. */
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);
    LOG(INFO) << "net inputs numbers is " << net->num_inputs();
    LOG(INFO) << "net outputs numbers is " << net->num_outputs();

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *net_input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "input data layer channels is  " << net_input_data_blobs->channels();
    LOG(INFO) << "input data layer width is  " << net_input_data_blobs->width();
    LOG(INFO) << "input data layer height is  " << net_input_data_blobs->height();
    
    size = net_input_data_blobs->channels()*net_input_data_blobs->width()*net_input_data_blobs->height();


    //uint64_t beginDataTime =  current_timestamp();
    //load image
    //im = load_image_color((char*)image_path.c_str(),0,0);
    cv::Mat im = cv::imread((char*)image_path.c_str()); 
    cv::cvtColor(im, im, CV_BGR2RGB);
    printf("func:%s,line:%d, im.h:%d,im.w:%d,im.c:%d\n", __FUNCTION__, __LINE__, im.rows, im.cols, im.channels());
    printf("func:%s,line:%d, network_im.h:%d,network_im.w:%d,network_im.c:%d, network_im.n:%d\n", __FUNCTION__, __LINE__, net_input_data_blobs->height(), net_input_data_blobs->width(), net_input_data_blobs->channels(), net_input_data_blobs->num());
    //sized = letterbox_image(im,net_input_data_blobs->width(),net_input_data_blobs->height());
    cv::Mat sized, f32Sized;
    cv::resize(im, sized, Size(net_input_data_blobs->width(), net_input_data_blobs->height()));
    printf("func:%s,line:%d, sized_im.h:%d,sized_im.w:%d,sized_im.c:%d\n", __FUNCTION__, __LINE__, sized.rows, sized.cols, sized.channels());
    sized.convertTo(f32Sized, CV_32FC3);
    #if 0
    cuda_push_array(net_input_data_blobs->mutable_gpu_data(),sized.data,size);
    #else
    cuda_push_array(net_input_data_blobs->mutable_cpu_data(),(float*)f32Sized.data,size);
    #endif

    //uint64_t endDataTime =  current_timestamp();
    //LOG(INFO) << "processing data operation avergae time is "
              //<< endDataTime - beginDataTime << " ms";
              
    //uint64_t startDetectTime = current_timestamp();
    // forward
    net->Forward();
    printf("func:%s, line:%d, num_outputs:%d\n", __FUNCTION__, __LINE__, net->num_outputs());
    for(int i =0;i<net->num_outputs();++i){
        blobs.push_back(net->output_blobs()[i]);
        printf("func:%s,line:%d, N:%d,C:%d,H:%d,W:%d\n", __FUNCTION__, __LINE__, net->output_blobs()[i]->num(), net->output_blobs()[i]->channels(),net->output_blobs()[i]->height(), net->output_blobs()[i]->width());
    }
    

    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
    	  Blob<float>* cls = blobs[_feat_stride_fpn.size() * i + 0];
    	  Blob<float>* reg = blobs[_feat_stride_fpn.size() * i + 1];
    	  Blob<float>* pts = blobs[_feat_stride_fpn.size() * i + 2];

        // get blob output
        //char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        //char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        //char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        //_extractor.extract(clsname, cls);
        //_extractor.extract(regname, reg);
        //_extractor.extract(ptsname, pts);

        printf("cls c:%d ,h:%d ,w:%d\n", cls->channels(), cls->height(), cls->width());
        printf("reg c:%d ,h:%d ,w:%d\n", reg->channels(), reg->height(), reg->width());
        printf("pts c:%d ,h:%d ,w:%d\n", pts->channels(), pts->height(), pts->width());

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        printf("stride %d, result size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }
    
     // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (sized, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
        for (int j = 0; j < result[i].pts.size(); ++j) {
        	cv::circle(sized, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
        }
    }
    result[0].print();

    //cv::imshow("img", im);
    cv::imwrite("result_112.jpg", sized);
    //cv::waitKey(0);
#endif
#if 0
    cv::Mat img = cv::imread("./images/test.jpg");
    if(!img.data)
    	printf("load error");


	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 300, 300);
    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) { 
    	ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
        for (int j = 0; j < result[i].pts.size(); ++j) {
        	cv::circle(img, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
        }
    }
    result[0].print();

    cv::imshow("img", img);
    cv::imwrite("result.jpg", img);
    cv::waitKey(0);
#endif
    return 0;
}

