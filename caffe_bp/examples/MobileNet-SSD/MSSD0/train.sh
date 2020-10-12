#!/bin/sh
if ! test -f MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
#../../../build/tools/caffe train -solver="solver_train.prototxt" \
#-weights="../mobilenet_iter_73000.caffemodel" \
#-gpu 1 
../../../build/tools/caffe train -solver="solver_train.prototxt" \
-snapshot="snapshot/mobilenet_iter_10000.solverstate" \
-gpu 0 
