#include <vector>
#include <iostream>
#include "caffe/layers/dice_layer.hpp"

namespace caffe {

template <typename Dtype>
void DiceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DiceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* score = bottom[0]->cpu_data();

  const int count = bottom[1]->count();

  for(int i = 0; i < count; ++i) {
      bottom[1]->mutable_cpu_diff()[i] = score[i] >= score[i+count] ? 0 : 1;
  }
  const Dtype* prediction = bottom[1]->cpu_diff();

  label_sum = caffe_cpu_asum(count, label);

  prediction_sum = caffe_cpu_asum(count, prediction);

  Dtype *intersection = bottom[1]->mutable_cpu_diff();
  caffe_mul(count, prediction, label, intersection);

  intersection_sum = caffe_cpu_asum(count, intersection);

  top[0]->mutable_cpu_data()[0] = 2.*intersection_sum/(label_sum + prediction_sum);

}

template <typename Dtype>
__global__ void ArgMax(const int n, const Dtype* score, Dtype* predictions) {
  CUDA_KERNEL_LOOP(i, n) {
      predictions[i] = score[i] >= score[i+n] ? 0 : 1;
  }
}

template <typename Dtype>
void DiceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << this->type() << bottom[1]->shape_string() << " " << bottom[0]->shape_string();

  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* score = bottom[0]->gpu_data();
  const int count = bottom[1]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  ArgMax<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->mutable_gpu_diff()
    );
  const Dtype* prediction = bottom[1]->gpu_diff();

  caffe_gpu_asum(count, prediction, &prediction_sum);
  LOG(INFO) << this->type() << " Prediction sum: " << prediction_sum;

  caffe_gpu_asum(count, label, &label_sum);
  LOG(INFO) << this->type() << " Label sum: " << label_sum;

  Dtype *intersection = bottom[1]->mutable_gpu_diff();
  caffe_gpu_mul(count, prediction, label, intersection);

  caffe_gpu_asum(count, intersection, &intersection_sum);
  LOG(INFO) << this->type() << " Intersection sum: " << intersection_sum;

  top[0]->mutable_cpu_data()[0] = 2.*intersection_sum/(label_sum + prediction_sum);
}

template <typename Dtype>
void DiceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


template <typename Dtype>
__global__ void DoBackward(const int labels_count, const Dtype* labels, const Dtype* predictions, Dtype* bottom_diff, const Dtype intersection, const Dtype unions) {
  CUDA_KERNEL_LOOP(j, labels_count) {
    Dtype grad = labels[j] / unions - intersection / (unions * unions);

    Dtype p0 = predictions[j];
    Dtype p1 = predictions[j + labels_count];

//    if (p0 == lablels[j])

    bottom_diff[j] = grad;
    bottom_diff[j + labels_count] = -grad;
  }
}

template <typename Dtype>
void DiceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."                                                                                                                                                                                               ;
  }
  if (propagate_down[1]) {
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DoBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      bottom[1]->mutable_gpu_diff(),
      intersection_sum,
      label_sum + prediction_sum
    );
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DiceLayer);
REGISTER_LAYER_CLASS(Dice);

}
