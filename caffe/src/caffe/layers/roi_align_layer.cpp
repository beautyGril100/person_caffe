// ------------------------------------------------------------------
// ROIAlignLayer written by Thanh-Toan Do
// Only a single value at each bin center is interpolated
// ------------------------------------------------------------------

#include <cfloat>
#include <stdio.h>
#include <math.h>
#include <float.h>

//#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h(); //7
  pooled_width_ = roi_align_param.pooled_w();  //7
  spatial_scale_ = roi_align_param.spatial_scale(); // 1/16
  /*****************added by hwzhu start****************/
  sampling_ratio_ = roi_align_param.sampling_ratio();
  /*****************added by hwzhu end  ****************/
  LOG(INFO) << "Spatial scale: " << spatial_scale_ << ", sampling_ratio_: " << sampling_ratio_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  channels_ = bottom[0]->channels(); // bottom[0]: conv5_3 (vgg16), // bottom[1]: rois
  height_ = bottom[0]->height(); // height of conv5_3
  width_ = bottom[0]->width();   // width of conv5_3
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // N x C x 7 x 7 where N is number of ROIS
//  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // Blob<int> max_idx_;
  //////////////////////////////////////////////////////////////////////////////
  max_idx_topleft.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_topright.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_bottomleft.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_bottomright.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  dh_ratio.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  dw_ratio.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  /////////////////////////////////////////////////////////////////////////////
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    Dtype roi_start_h,
    Dtype roi_start_w,
    Dtype bin_size_h,
    Dtype bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const Dtype yy = roi_start_h + ph * bin_size_h +
            static_cast<Dtype>(iy + .5f) * bin_size_h /
                static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const Dtype xx = roi_start_w + pw * bin_size_w +
              static_cast<Dtype>(ix + .5f) * bin_size_w /
                  static_cast<Dtype>(roi_bin_grid_w);

          Dtype x = xx;
          Dtype y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (Dtype)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (Dtype)x_low;
          } else {
            x_high = x_low + 1;
          }

          Dtype ly = y - y_low;
          Dtype lx = x - x_low;
          Dtype hy = 1. - ly, hx = 1. - lx;
          Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data(); //conv5_3
  const Dtype* bottom_rois = bottom[1]->cpu_data(); // rois

  int num_rois = bottom[1]->num(); // Number of ROIs
  int batch_size = bottom[0]->num(); // number of image = 1
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
//int* argmax_data = max_idx_.mutable_cpu_data();
//caffe_set(top_count, -1, argmax_data);
  /////////////////////////////////////////////////////////////////
  int* argmax_data_topleft = max_idx_topleft.mutable_cpu_data();
  int* argmax_data_topright = max_idx_topright.mutable_cpu_data();
  int* argmax_data_bottomleft = max_idx_bottomleft.mutable_cpu_data();
  int* argmax_data_bottomright = max_idx_bottomright.mutable_cpu_data();
  float* dh_ratio_data = dh_ratio.mutable_cpu_data();
  float* dw_ratio_data = dw_ratio.mutable_cpu_data();

  caffe_set(top_count, -1, argmax_data_topleft);
  caffe_set(top_count, -1, argmax_data_topright);
  caffe_set(top_count, -1, argmax_data_bottomleft);
  caffe_set(top_count, -1, argmax_data_bottomright);
  caffe_set(top_count, (float)(0), dh_ratio_data);
  caffe_set(top_count, (float)(0), dw_ratio_data);
  /////////////////////////////////////////////////////////////////


  // For each ROI R = [batch_index x1 y1 x2 y2]:
  for (int n = 0; n < num_rois; ++n) //scan each roi
  {
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = (bottom_rois[1] * spatial_scale_); // x1*1/16. Because (x1,y1,x2,y2) is coordinate in input image, divide by 16 --> coordinate in conv5_3
    float roi_start_h = (bottom_rois[2] * spatial_scale_); // y1*1/16
    float roi_end_w = (bottom_rois[3] * spatial_scale_);  // x2*1/16
    float roi_end_h = (bottom_rois[4] * spatial_scale_);  // y2*1/16
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1); // roi_height: height of roi (w.r.t. conv5_3 size)
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1); // roi_width: width of roi (w.r.t. conv5_3 size)
    float bin_size_h = (roi_height) / ((float)(pooled_height_)); // bin_size_h = roi_height/7 = height stride (w.r.t. conv5_3 size)
    float bin_size_w = (roi_width) / ((float)(pooled_width_)); // bin_size_w = roi_width/7 = width stride (w.r.t. conv5_3 size)
	
	/*************************************added by hwzhu start *************************************/
	// We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio_ > 0)
        ? sampling_ratio_
        : ceil(roi_height / pooled_height_); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width_);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const Dtype count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4
	// we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width_ * pooled_height_);
    pre_calc_for_bilinear_interpolate(
        height_,
        width_,
        pooled_height_,
        pooled_width_,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);
	/************************************added by hwzhu end*****************************************/

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c)  // for each channel in conv5_3
    {
	  /*************added by hwzhu start*****************/
	  int pre_calc_index = 0;
	  /*************added by hwzhu end  *****************/
      // for each bin in 7x7 grid
      for (int ph = 0; ph < pooled_height_; ++ph) // pooled_height_ = 7
      {
        for (int pw = 0; pw < pooled_width_; ++pw)  // pooled_width_ = 7
        {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          // (hstart, wstart), (hend, wend): sub window in conv5_3 in which max value is got
          float hstart = ((float)(ph))* bin_size_h;
          float wstart = ((float)(pw))* bin_size_w;
          float hend = ((float)(ph + 1))* bin_size_h;
          float wend = ((float)(pw + 1))* bin_size_w;

//          hstart = fminf(fmaxf(hstart + roi_start_h, 0), height_);// hstart = hstart + roi_start_h:   coordinate of subwindow in conv5_3
//          hend = fminf(fmaxf(hend + roi_start_h, 0), height_);    // hend = hend + roi_start_h:       coordinate of subwindow in conv5_3
//          wstart = fminf(fmaxf(wstart + roi_start_w, 0), width_); // wstart = wstart + roi_start_w:   coordinate of subwindow in conv5_3
//          wend = fminf(fmaxf(wend + roi_start_w, 0), width_);     // wend = wend + roi_start_w:       coordinate of subwindow in conv5_3
//          bool is_empty = (hend <= hstart) || (wend <= wstart);
          hstart = fminf(fmaxf(hstart + roi_start_h, 0), height_ -1);
          hend = fminf(fmaxf(hend + roi_start_h, 0), height_ -1);
          wstart = fminf(fmaxf(wstart + roi_start_w, 0), width_ -1);
          wend = fminf(fmaxf(wend + roi_start_w, 0), width_ -1);
          bool is_empty = (hend < hstart) || (wend < wstart);

          const int pool_index = ph * pooled_width_ + pw; // bin id (in 7x7 grid)
          if (is_empty)
          {
            top_data[pool_index] = 0;
//            argmax_data[pool_index] = -1;
            argmax_data_topleft[pool_index] = -1;
            argmax_data_topright[pool_index] = -1;
            argmax_data_bottomleft[pool_index] = -1;
            argmax_data_bottomright[pool_index] = -1;
            dh_ratio_data[pool_index] = 0;
            dw_ratio_data[pool_index] = 0;
			/*************added by hwzhu start*****************/
			pre_calc_index += 1;
			/*************added by hwzhu end  *****************/
          }

//          // find max value in sub window (wstart, hstart, wstart, wend). (wstart, hstart, wstart, wend) is coordinate in conv5_3
//          for (int h = hstart; h < hend; ++h)
//          {
//            for (int w = wstart; w < wend; ++w)
//            {
//              const int index = h * width_ + w; // index in conv5_3
//              if (batch_data[index] > top_data[pool_index])
//              {
//                top_data[pool_index] = batch_data[index];
//                argmax_data[pool_index] = index; // store the index where max value is got; this index coordinate is in conv5_3
//              }
//            }
//          }

          else
          {
			  /**********************************added by hwzhu start**************************************/
			  Dtype output_val = 0.;
			  for (int iy = 0; iy < roi_bin_grid_h; iy++) {
			    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
				  PreCalc pc = pre_calc[pre_calc_index];
				  Dtype tmp_output_val = pc.w1 * batch_data[pc.pos1] +
					      pc.w2 * batch_data[pc.pos2] +
					      pc.w3 * batch_data[pc.pos3] + 
						  pc.w4 * batch_data[pc.pos4];
				  if (tmp_output_val > output_val){
					  output_val = tmp_output_val;
				  }

				  pre_calc_index += 1;
				}
			  }
			  //output_val /= count;	//average pooling
			  top_data[pool_index] = output_val;
			  /**********************************added by hwzhu end  **************************************/
#if 0	//commented by hwzhu
			  // find the center of sub window (w.r.t. conv5_3 size)
			  float centerx = (wstart + wend)/2;
			  float centery = (hstart + hend)/2;

			  // 4 nearest around the center
//			  int cy_top = 	min(  max(  (int)(floor(centery)), 0)  ,height_);
//			  int cy_bottom = min(  max(  (int)(ceil(centery)), 0), height_);
//			  int cx_left = 	min(  max(  (int)(floor(centerx)), 0), width_);
//			  int cx_right = 	min(  max(  (int)(ceil(centerx)), 0), width_);
			  int cy_top = 	min(  max(  static_cast<int>(floor(centery)), 0)  ,height_-1);
			  int cy_bottom = min(  max(  static_cast<int>(ceil(centery)), 0), height_-1);
			  int cx_left = 	min(  max(  static_cast<int>(floor(centerx)), 0), width_-1);
			  int cx_right = 	min(  max(  static_cast<int>(ceil(centerx)), 0), width_-1);
			  // find indexes of 4 nearest around the center
			  int topleft = cy_top * width_ + cx_left;
			  int topright = cy_top * width_ + cx_right;
			  int bottomleft = cy_bottom * width_ + cx_left;
			  int bottomright = cy_bottom * width_ + cx_right;
			  // bilinear interpolate bin value using the 4 around nearest
			  float y_ratio =  centery - (float)(cy_top); // vertical distance to topleft
			  float x_ratio =  centerx - (float)(cx_left); // horizontal distance to topleft

//			  printf("y_ratio: %f", y_ratio);
//			  printf("x_ratio: %f", x_ratio);

			  top_data[pool_index] = batch_data[topleft] * (1-y_ratio) * (1-x_ratio)
								  +  batch_data[topright] * (1-y_ratio) * (x_ratio)
								  +  batch_data[bottomleft] * (y_ratio) * (1 - x_ratio)
								  +  batch_data[bottomright] * (y_ratio) * (x_ratio);

			  argmax_data_topleft[pool_index] = topleft; // store topleft index
			  argmax_data_topright[pool_index] = topright;
			  argmax_data_bottomleft[pool_index] = bottomleft;
			  argmax_data_bottomright[pool_index] = bottomright; //store bottom right index
			  dh_ratio_data[pool_index] = y_ratio;
			  dw_ratio_data[pool_index] = x_ratio;
#endif
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
//      argmax_data += max_idx_.offset(0, 1);
      argmax_data_topleft += max_idx_topleft.offset(0, 1);
      argmax_data_topright += max_idx_topright.offset(0, 1);
      argmax_data_bottomleft += max_idx_bottomleft.offset(0, 1);
      argmax_data_bottomright += max_idx_bottomright.offset(0, 1);
      dh_ratio_data += dh_ratio.offset(0, 1);
      dw_ratio_data += dw_ratio.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}


template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
