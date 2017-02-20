#include <cfloat>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dice_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DiceLayerTest : public MultiDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;
 protected:
  DiceLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    shape.resize(3);
    shape[0] = 3;
    shape[1] = 2;
    shape[2] = 25;
    blob_bottom_data_->Reshape(shape);
    shape[1] = 1;
    blob_bottom_label_->Reshape(shape);

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DiceLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }

  vector<int> shape;

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DiceLayerTest, TestDtypesAndDevices);

  TYPED_TEST(DiceLayerTest, TestDice) {
    typedef typename TypeParam::Dtype Dtype;
    Dtype *label_data = this->blob_bottom_label_->mutable_cpu_data();

    int positive_labels = 25;
    for(int i = 0; i < positive_labels; ++i) {
        label_data[i] = Dtype(1.);
        label_data[i+this->shape[2]] = Dtype(1.);
        label_data[i+2*this->shape[2]] = Dtype(1.);
    }

    Dtype *data = this->blob_bottom_data_->mutable_cpu_data();

    int positive_predictions = 75;
    for(int i = 0; i < positive_predictions; ++i) {
        data[i+this->blob_bottom_label_->count()] = Dtype(3.5);
    }

    LayerParameter layer_param;
    DiceLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Now, check values
    Dtype intersection = Dtype(std::min(positive_labels*3, positive_predictions));
    positive_labels *= 3;
    Dtype unions = Dtype(positive_labels + positive_predictions);

    EXPECT_NEAR(Dtype(2.)*intersection/unions, this->blob_top_->cpu_data()[0], 1e-5);
  }

}
