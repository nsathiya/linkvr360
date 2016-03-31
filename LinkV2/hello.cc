// hello.cc
#include <node.h>
#include <iostream>
#include "Test.h"
#include <node_buffer.h>
#include <nan.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/nonfree/nonfree.hpp"

namespace demo {

using v8::FunctionCallbackInfo;
using v8::Function;
using v8::Exception;
using v8::Isolate;
using v8::Local;
using v8::Object;
using v8::String;
using v8::Value;
using v8::Integer;

void Method(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Test hello = Test(28);

  if (args.Length() < 1) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Wrong number of arguments")));
    return;
  }
  cv::Mat mat1, mat2, mat3;
  if (node::Buffer::HasInstance(args[0])) {
      uint8_t *buf = (uint8_t *) node::Buffer::Data(args[0]->ToObject());
      unsigned len = node::Buffer::Length(args[0]->ToObject());

      cv::Mat *mbuf = new cv::Mat(len, 1, CV_64FC1, buf);
      mat1 = cv::imdecode(*mbuf, -1); 
  }

  if (node::Buffer::HasInstance(args[1])) {
      uint8_t *buf = (uint8_t *) node::Buffer::Data(args[1]->ToObject());
      unsigned len = node::Buffer::Length(args[1]->ToObject());

      cv::Mat *mbuf = new cv::Mat(len, 1, CV_64FC1, buf);
      mat2 = cv::imdecode(*mbuf, -1); 
  }
  if (node::Buffer::HasInstance(args[2])) {
      uint8_t *buf = (uint8_t *) node::Buffer::Data(args[2]->ToObject());
      unsigned len = node::Buffer::Length(args[2]->ToObject());

      cv::Mat *mbuf = new cv::Mat(len, 1, CV_64FC1, buf);
      mat3 = cv::imdecode(*mbuf, -1); 
  }

  std::vector<uchar> rbuff(0);
  cv::Mat result = hello.getWorld(mat1, mat2, mat3);
  //std::cout << result << std::endl;
  cv::imencode(".jpg", result, rbuff);
  std::cout << "encoded" << std::endl;
  // This is Buffer that actually makes heap-allocated raw binary available
  // to userland code.
  Local<Object> buff = Nan::NewBuffer(rbuff.size()).ToLocalChecked();
  std::cout << "made a local buffer" << std::endl;
  // Buffer:Data gives us a yummy void* pointer to play with to our hearts
  // content.
  uchar* data = (uchar*) node::Buffer::Data(buff);
  memcpy(data, &rbuff[0], rbuff.size());
  std::cout << "made a 'node' buffer and copied to uchar block" << std::endl;
  // Now we need to create the JS version of the Buffer I was telling you about.
  // To do that we need to actually pull it from the execution context.
  // First step is to get a handle to the global object.
  Local<Object> globalObj = Nan::GetCurrentContext()->Global();
  std::cout << "grabbed the global context" << std::endl;
  // Now we need to grab the Buffer constructor function.
  //v8::Local<v8::Function> bufferConstructor = v8::Local<v8::Function>::Cast(globalObj->Get(v8::String::New("Buffer")));
  Local<Function> bufferConstructor = Local<Function>::Cast(globalObj->Get(Nan::New<String>("Buffer").ToLocalChecked()));
  std::cout << "created a buffer constructor from global context" << std::endl;
  // Great. We can use this constructor function to allocate new Buffers.
  // Let's do that now. First we need to provide the correct arguments.
  // First argument is the JS object Handle for the SlowBuffer.
  // Second arg is the length of the SlowBuffer.
  // Third arg is the offset in the SlowBuffer we want the .. "Fast"Buffer to start at.
  //v8::Handle<v8::Value> constructorArgs[3] = { slowBuffer->handle_, v8::Integer::New(length), v8::Integer::New(0) };
  Local<Value> constructorArgs[3] = {buff, Nan::New<Integer>((unsigned)rbuff.size()), Nan::New<Integer>(0)};
  std::cout << "set up arguments for constructor" << std::endl;

  // Now we have our constructor, and our constructor args. Let's create the 
  // damn Buffer already!
  Local<Object> actualBuffer = bufferConstructor->NewInstance(3, constructorArgs);
  std::cout << "made buffer with full data" << std::endl;

  std::cout << "returning value" << std::endl;
  //args.GetReturnValue().Set(Integer::New(isolate,hello.getWorld()));
    args.GetReturnValue().Set(actualBuffer);
}

void init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "hello", Method);
}

NODE_MODULE(addon, init)

}  // namespace demo