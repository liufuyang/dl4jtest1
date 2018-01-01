# A simple deeplearning4j test code

A few issues:

## About hand calculated accuracy

When switching to using CUDA, the output shows hand calculated acc looks different with results calculated with
build in Evaluation class.

Such output examples are:
```
2018-01-01T15:27:10.682 [main] INFO  DL4JTest1 - Evaluation on test data - [Epoch 1] [AccHand: 0.302, Accuracy: 0.302, P: 0.250, R: 0.302, F1: 0.464]
2018-01-01T15:27:11.367 [main] INFO  DL4JTest1 - Evaluation on test data - [Epoch 2] [AccHand: 0.277, Accuracy: 0.297, P: 0.250, R: 0.297, F1: 0.458]
2018-01-01T15:27:11.901 [main] INFO  DL4JTest1 - Evaluation on test data - [Epoch 3] [AccHand: 0.290, Accuracy: 0.305, P: 0.250, R: 0.305, F1: 0.467]
2018-01-01T15:27:12.409 [main] INFO  DL4JTest1 - Evaluation on test data - [Epoch 4] [AccHand: 0.290, Accuracy: 0.299, P: 0.250, R: 0.299, F1: 0.460]
```

However this issue not seen on using CPU to do the calculation.

-------

## About CUDA issue

In the code line 110, when switching this
```
if (labels_s.shape()[0] > 10000) break;
```

to this
```
if (labels_s.shape()[0] > 20000) break;
```

A CUDA error happens:
```
/home/fuyang/Software/Java/jdk-9.0.1/bin/java -Xms2G -Xmx6G -Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=9G -javaagent:/home/fuyang/Software/idea-IC-173.4127.27/lib/idea_rt.jar=42487:/home/fuyang/Software/idea-IC-173.4127.27/bin -Dfile.encoding=UTF-8 -classpath /home/fuyang/Workspace/dl4jtest1/target/classes:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0-platform/0.9.1/nd4j-cuda-8.0-platform-0.9.1.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda-platform/8.0-6.0-1.3/cuda-platform-8.0-6.0-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda/8.0-6.0-1.3/cuda-8.0-6.0-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda/8.0-6.0-1.3/cuda-8.0-6.0-1.3-linux-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda/8.0-6.0-1.3/cuda-8.0-6.0-1.3-linux-ppc64le.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda/8.0-6.0-1.3/cuda-8.0-6.0-1.3-macosx-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/cuda/8.0-6.0-1.3/cuda-8.0-6.0-1.3-windows-x86_64.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp/1.3.3/javacpp-1.3.3.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-x86_64.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-macosx-x86_64.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-windows-x86_64.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-ppc64le.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-core/0.9.1/deeplearning4j-core-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/nearestneighbor-core/0.9.1/nearestneighbor-core-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-modelimport/0.9.1/deeplearning4j-modelimport-0.9.1.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5-platform/1.10.0-patch1-1.3/hdf5-platform-1.10.0-patch1-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-linux-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-linux-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-linux-ppc64le.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-macosx-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-windows-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/hdf5/1.10.0-patch1-1.3/hdf5-1.10.0-patch1-1.3-windows-x86_64.jar:/home/fuyang/.m2/repository/org/slf4j/slf4j-api/1.7.12/slf4j-api-1.7.12.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-nn/0.9.1/deeplearning4j-nn-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-common/0.9.1/nd4j-common-0.9.1.jar:/home/fuyang/.m2/repository/com/github/stephenc/findbugs/findbugs-annotations/1.3.9-1/findbugs-annotations-1.3.9-1.jar:/home/fuyang/.m2/repository/org/apache/commons/commons-math3/3.4.1/commons-math3-3.4.1.jar:/home/fuyang/.m2/repository/commons-io/commons-io/2.4/commons-io-2.4.jar:/home/fuyang/.m2/repository/org/apache/commons/commons-compress/1.8/commons-compress-1.8.jar:/home/fuyang/.m2/repository/org/tukaani/xz/1.5/xz-1.5.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-api/0.9.1/nd4j-api-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-buffer/0.9.1/nd4j-buffer-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-context/0.9.1/nd4j-context-0.9.1.jar:/home/fuyang/.m2/repository/net/ericaro/neoitertools/1.0.0/neoitertools-1.0.0.jar:/home/fuyang/.m2/repository/junit/junit/4.8.2/junit-4.8.2.jar:/home/fuyang/.m2/repository/org/reflections/reflections/0.9.10/reflections-0.9.10.jar:/home/fuyang/.m2/repository/com/google/guava/guava/15.0/guava-15.0.jar:/home/fuyang/.m2/repository/org/javassist/javassist/3.19.0-GA/javassist-3.19.0-GA.jar:/home/fuyang/.m2/repository/com/google/code/findbugs/annotations/2.0.1/annotations-2.0.1.jar:/home/fuyang/.m2/repository/org/apache/commons/commons-lang3/3.4/commons-lang3-3.4.jar:/home/fuyang/.m2/repository/org/nd4j/jackson/0.9.1/jackson-0.9.1.jar:/home/fuyang/.m2/repository/org/yaml/snakeyaml/1.12/snakeyaml-1.12.jar:/home/fuyang/.m2/repository/org/codehaus/woodstox/stax2-api/3.1.4/stax2-api-3.1.4.jar:/home/fuyang/.m2/repository/joda-time/joda-time/2.2/joda-time-2.2.jar:/home/fuyang/.m2/repository/org/projectlombok/lombok/1.16.16/lombok-1.16.16.jar:/home/fuyang/.m2/repository/org/datavec/datavec-api/0.9.1/datavec-api-0.9.1.jar:/home/fuyang/.m2/repository/org/freemarker/freemarker/2.3.23/freemarker-2.3.23.jar:/home/fuyang/.m2/repository/com/clearspring/analytics/stream/2.7.0/stream-2.7.0.jar:/home/fuyang/.m2/repository/it/unimi/dsi/fastutil/6.5.7/fastutil-6.5.7.jar:/home/fuyang/.m2/repository/net/sf/opencsv/opencsv/2.3/opencsv-2.3.jar:/home/fuyang/.m2/repository/org/datavec/datavec-data-image/0.9.1/datavec-data-image-0.9.1.jar:/home/fuyang/.m2/repository/com/github/jai-imageio/jai-imageio-core/1.3.0/jai-imageio-core-1.3.0.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-jpeg/3.1.1/imageio-jpeg-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-core/3.1.1/imageio-core-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-metadata/3.1.1/imageio-metadata-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/common/common-lang/3.1.1/common-lang-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/common/common-io/3.1.1/common-io-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/common/common-image/3.1.1/common-image-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-tiff/3.1.1/imageio-tiff-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-psd/3.1.1/imageio-psd-3.1.1.jar:/home/fuyang/.m2/repository/com/twelvemonkeys/imageio/imageio-bmp/3.1.1/imageio-bmp-3.1.1.jar:/home/fuyang/.m2/repository/org/bytedeco/javacv/1.3.3/javacv-1.3.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/ffmpeg/3.2.1-1.3/ffmpeg-3.2.1-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/flycapture/2.9.3.43-1.3/flycapture-2.9.3.43-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/libdc1394/2.2.4-1.3/libdc1394-2.2.4-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/libfreenect/0.5.3-1.3/libfreenect-0.5.3-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/libfreenect2/0.2.0-1.3/libfreenect2-0.2.0-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/librealsense/1.9.6-1.3/librealsense-1.9.6-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/videoinput/0.200-1.3/videoinput-0.200-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/artoolkitplus/2.3.1-1.3/artoolkitplus-2.3.1-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/flandmark/1.07-1.3/flandmark-1.07-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv-platform/3.2.0-1.3/opencv-platform-3.2.0-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-android-arm.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-android-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-linux-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-linux-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-linux-armhf.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-linux-ppc64le.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-macosx-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-windows-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/opencv/3.2.0-1.3/opencv-3.2.0-1.3-windows-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica-platform/1.73-1.3/leptonica-platform-1.73-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-android-arm.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-android-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-linux-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-linux-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-linux-armhf.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-linux-ppc64le.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-macosx-x86_64.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-windows-x86.jar:/home/fuyang/.m2/repository/org/bytedeco/javacpp-presets/leptonica/1.73-1.3/leptonica-1.73-1.3-windows-x86_64.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-ui-components/0.9.1/deeplearning4j-ui-components-0.9.1.jar:/home/fuyang/.m2/repository/commons-codec/commons-codec/1.10/commons-codec-1.10.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-nlp/0.9.1/deeplearning4j-nlp-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-native-api/0.9.1/nd4j-native-api-0.9.1.jar:/home/fuyang/.m2/repository/commons-lang/commons-lang/2.6/commons-lang-2.6.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson/0.9.1/nd4j-jackson-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-base64/0.9.1/nd4j-base64-0.9.1.jar:/home/fuyang/.m2/repository/commons-net/commons-net/3.1/commons-net-3.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-zoo/0.9.1/deeplearning4j-zoo-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-ui_2.11/0.9.1/deeplearning4j-ui_2.11-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-play_2.11/0.9.1/deeplearning4j-play_2.11-0.9.1.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-java_2.11/2.4.6/play-java_2.11-2.4.6.jar:/home/fuyang/.m2/repository/org/scala-lang/scala-library/2.11.6/scala-library-2.11.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play_2.11/2.4.6/play_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/build-link/2.4.6/build-link-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-exceptions/2.4.6/play-exceptions-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-iteratees_2.11/2.4.6/play-iteratees_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/config/1.3.0/config-1.3.0.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-json_2.11/2.4.6/play-json_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-functional_2.11/2.4.6/play-functional_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-datacommons_2.11/2.4.6/play-datacommons_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-netty-utils/2.4.6/play-netty-utils-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/twirl-api_2.11/1.1.1/twirl-api_2.11-1.1.1.jar:/home/fuyang/.m2/repository/org/scala-lang/modules/scala-xml_2.11/1.0.1/scala-xml_2.11-1.0.1.jar:/home/fuyang/.m2/repository/org/scala-lang/modules/scala-parser-combinators_2.11/1.0.1/scala-parser-combinators_2.11-1.0.1.jar:/home/fuyang/.m2/repository/org/slf4j/jul-to-slf4j/1.7.12/jul-to-slf4j-1.7.12.jar:/home/fuyang/.m2/repository/org/slf4j/jcl-over-slf4j/1.7.12/jcl-over-slf4j-1.7.12.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-actor_2.11/2.3.13/akka-actor_2.11-2.3.13.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-slf4j_2.11/2.3.13/akka-slf4j_2.11-2.3.13.jar:/home/fuyang/.m2/repository/org/scala-stm/scala-stm_2.11/0.7/scala-stm_2.11-0.7.jar:/home/fuyang/.m2/repository/org/joda/joda-convert/1.7/joda-convert-1.7.jar:/home/fuyang/.m2/repository/xerces/xercesImpl/2.11.0/xercesImpl-2.11.0.jar:/home/fuyang/.m2/repository/xml-apis/xml-apis/1.4.01/xml-apis-1.4.01.jar:/home/fuyang/.m2/repository/javax/transaction/jta/1.1/jta-1.1.jar:/home/fuyang/.m2/repository/com/google/inject/guice/4.0/guice-4.0.jar:/home/fuyang/.m2/repository/javax/inject/javax.inject/1/javax.inject-1.jar:/home/fuyang/.m2/repository/aopalliance/aopalliance/1.0/aopalliance-1.0.jar:/home/fuyang/.m2/repository/com/google/inject/extensions/guice-assistedinject/4.0/guice-assistedinject-4.0.jar:/home/fuyang/.m2/repository/org/scala-lang/modules/scala-java8-compat_2.11/0.3.0/scala-java8-compat_2.11-0.3.0.jar:/home/fuyang/.m2/repository/org/hibernate/hibernate-validator/5.0.3.Final/hibernate-validator-5.0.3.Final.jar:/home/fuyang/.m2/repository/javax/validation/validation-api/1.1.0.Final/validation-api-1.1.0.Final.jar:/home/fuyang/.m2/repository/com/fasterxml/classmate/1.0.0/classmate-1.0.0.jar:/home/fuyang/.m2/repository/org/jboss/logging/jboss-logging/3.2.1.Final/jboss-logging-3.2.1.Final.jar:/home/fuyang/.m2/repository/org/springframework/spring-context/4.1.6.RELEASE/spring-context-4.1.6.RELEASE.jar:/home/fuyang/.m2/repository/org/springframework/spring-core/4.1.6.RELEASE/spring-core-4.1.6.RELEASE.jar:/home/fuyang/.m2/repository/org/springframework/spring-beans/4.1.6.RELEASE/spring-beans-4.1.6.RELEASE.jar:/home/fuyang/.m2/repository/net/jodah/typetools/0.4.3/typetools-0.4.3.jar:/home/fuyang/.m2/repository/org/apache/tomcat/tomcat-servlet-api/8.0.21/tomcat-servlet-api-8.0.21.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-netty-server_2.11/2.4.6/play-netty-server_2.11-2.4.6.jar:/home/fuyang/.m2/repository/com/typesafe/play/play-server_2.11/2.4.6/play-server_2.11-2.4.6.jar:/home/fuyang/.m2/repository/io/netty/netty/3.10.4.Final/netty-3.10.4.Final.jar:/home/fuyang/.m2/repository/com/typesafe/netty/netty-http-pipelining/1.1.4/netty-http-pipelining-1.1.4.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-contrib_2.11/2.3.13/akka-contrib_2.11-2.3.13.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-remote_2.11/2.3.13/akka-remote_2.11-2.3.13.jar:/home/fuyang/.m2/repository/com/google/protobuf/protobuf-java/2.5.0/protobuf-java-2.5.0.jar:/home/fuyang/.m2/repository/org/uncommons/maths/uncommons-maths/1.2.2a/uncommons-maths-1.2.2a.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-persistence-experimental_2.11/2.3.13/akka-persistence-experimental_2.11-2.3.13.jar:/home/fuyang/.m2/repository/org/iq80/leveldb/leveldb/0.5/leveldb-0.5.jar:/home/fuyang/.m2/repository/org/iq80/leveldb/leveldb-api/0.5/leveldb-api-0.5.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-all/1.7/leveldbjni-all-1.7.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni/1.7/leveldbjni-1.7.jar:/home/fuyang/.m2/repository/org/fusesource/hawtjni/hawtjni-runtime/1.8/hawtjni-runtime-1.8.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-osx/1.5/leveldbjni-osx-1.5.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-linux32/1.5/leveldbjni-linux32-1.5.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-linux64/1.5/leveldbjni-linux64-1.5.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-win32/1.5/leveldbjni-win32-1.5.jar:/home/fuyang/.m2/repository/org/fusesource/leveldbjni/leveldbjni-win64/1.5/leveldbjni-win64-1.5.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/core/jackson-core/2.4.4/jackson-core-2.4.4.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/core/jackson-databind/2.4.4/jackson-databind-2.4.4.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/core/jackson-annotations/2.4.4/jackson-annotations-2.4.4.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/module/jackson-module-scala_2.11/2.4.4/jackson-module-scala_2.11-2.4.4.jar:/home/fuyang/.m2/repository/org/scala-lang/scala-reflect/2.11.2/scala-reflect-2.11.2.jar:/home/fuyang/.m2/repository/com/thoughtworks/paranamer/paranamer/2.6/paranamer-2.6.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/datatype/jackson-datatype-jdk8/2.4.4/jackson-datatype-jdk8-2.4.4.jar:/home/fuyang/.m2/repository/com/fasterxml/jackson/datatype/jackson-datatype-jsr310/2.4.4/jackson-datatype-jsr310-2.4.4.jar:/home/fuyang/.m2/repository/com/typesafe/akka/akka-cluster_2.11/2.3.13/akka-cluster_2.11-2.3.13.jar:/home/fuyang/.m2/repository/javax/ws/rs/javax.ws.rs-api/2.0/javax.ws.rs-api-2.0.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-ui-model/0.9.1/deeplearning4j-ui-model-0.9.1.jar:/home/fuyang/.m2/repository/org/agrona/Agrona/0.5.4/Agrona-0.5.4.jar:/home/fuyang/.m2/repository/org/mapdb/mapdb/3.0.5/mapdb-3.0.5.jar:/home/fuyang/.m2/repository/org/jetbrains/kotlin/kotlin-stdlib/1.0.7/kotlin-stdlib-1.0.7.jar:/home/fuyang/.m2/repository/org/jetbrains/kotlin/kotlin-runtime/1.0.7/kotlin-runtime-1.0.7.jar:/home/fuyang/.m2/repository/org/eclipse/collections/eclipse-collections-api/7.1.1/eclipse-collections-api-7.1.1.jar:/home/fuyang/.m2/repository/net/jcip/jcip-annotations/1.0/jcip-annotations-1.0.jar:/home/fuyang/.m2/repository/org/eclipse/collections/eclipse-collections/7.1.1/eclipse-collections-7.1.1.jar:/home/fuyang/.m2/repository/org/eclipse/collections/eclipse-collections-forkjoin/7.1.1/eclipse-collections-forkjoin-7.1.1.jar:/home/fuyang/.m2/repository/net/jpountz/lz4/lz4/1.3.0/lz4-1.3.0.jar:/home/fuyang/.m2/repository/org/mapdb/elsa/3.0.0-M5/elsa-3.0.0-M5.jar:/home/fuyang/.m2/repository/org/xerial/sqlite-jdbc/3.15.1/sqlite-jdbc-3.15.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-ui-resources/0.9.1/deeplearning4j-ui-resources-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/deeplearning4j-parallel-wrapper_2.11/0.9.1/deeplearning4j-parallel-wrapper_2.11-0.9.1.jar:/home/fuyang/.m2/repository/com/beust/jcommander/1.27/jcommander-1.27.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server/0.9.1/nd4j-parameter-server-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-model/0.9.1/nd4j-parameter-server-model-0.9.1.jar:/home/fuyang/.m2/repository/com/mashape/unirest/unirest-java/1.4.9/unirest-java-1.4.9.jar:/home/fuyang/.m2/repository/org/apache/httpcomponents/httpasyncclient/4.1.1/httpasyncclient-4.1.1.jar:/home/fuyang/.m2/repository/org/apache/httpcomponents/httpcore-nio/4.4.4/httpcore-nio-4.4.4.jar:/home/fuyang/.m2/repository/org/apache/httpcomponents/httpmime/4.5.2/httpmime-4.5.2.jar:/home/fuyang/.m2/repository/org/json/json/20160212/json-20160212.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-aeron/0.9.1/nd4j-aeron-0.9.1.jar:/home/fuyang/.m2/repository/io/aeron/aeron-all/1.0.4/aeron-all-1.0.4.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-client/0.9.1/nd4j-parameter-server-client-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/arbiter-deeplearning4j/0.9.1/arbiter-deeplearning4j-0.9.1.jar:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson-reflectionloader/0.9.1/nd4j-jackson-reflectionloader-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/arbiter-core/0.9.1/arbiter-core-0.9.1.jar:/home/fuyang/.m2/repository/org/deeplearning4j/arbiter-ui_2.11/0.9.1/arbiter-ui_2.11-0.9.1.jar:/home/fuyang/.m2/repository/org/datavec/datavec-data-codec/0.9.1/datavec-data-codec-0.9.1.jar:/home/fuyang/.m2/repository/org/jcodec/jcodec/0.1.5/jcodec-0.1.5.jar:/home/fuyang/.m2/repository/jfree/jfreechart/1.0.13/jfreechart-1.0.13.jar:/home/fuyang/.m2/repository/jfree/jcommon/1.0.16/jcommon-1.0.16.jar:/home/fuyang/.m2/repository/org/jfree/jcommon/1.0.23/jcommon-1.0.23.jar:/home/fuyang/.m2/repository/org/apache/httpcomponents/httpclient/4.3.5/httpclient-4.3.5.jar:/home/fuyang/.m2/repository/org/apache/httpcomponents/httpcore/4.3.2/httpcore-4.3.2.jar:/home/fuyang/.m2/repository/commons-logging/commons-logging/1.1.3/commons-logging-1.1.3.jar:/home/fuyang/.m2/repository/ch/qos/logback/logback-classic/1.1.7/logback-classic-1.1.7.jar:/home/fuyang/.m2/repository/ch/qos/logback/logback-core/1.1.7/logback-core-1.1.7.jar DL4JTest1
2018-01-01T15:40:29.453 [main] INFO  org.nd4j.linalg.factory.Nd4jBackend - Loaded [JCublasBackend] backend
2018-01-01T15:40:29.534 [main] WARN  org.reflections.Reflections - given scan urls are empty. set urls in the configuration
2018-01-01T15:40:30.629 [main] INFO  org.nd4j.nativeblas.NativeOpsHolder - Number of threads used for NativeOps: 32
2018-01-01T15:40:30.963 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.977 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.978 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.978 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.979 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.979 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.980 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.981 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.981 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.982 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.982 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.983 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.984 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.984 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.985 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.986 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.989 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.990 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.991 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.992 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.992 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.993 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.998 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.998 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:30.999 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.000 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.001 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.001 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.002 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.003 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.004 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.005 [main] DEBUG o.n.j.a.c.impl.BasicContextPool - Creating new stream for thread: [1], device: [0]...
2018-01-01T15:40:31.007 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [19] to device [0], out of [1] devices...
2018-01-01T15:40:31.015 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [20] to device [0], out of [1] devices...
2018-01-01T15:40:31.016 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [21] to device [0], out of [1] devices...
2018-01-01T15:40:31.019 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [22] to device [0], out of [1] devices...
2018-01-01T15:40:31.019 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [23] to device [0], out of [1] devices...
2018-01-01T15:40:31.022 [main] DEBUG o.n.j.c.CudaAffinityManager - Manually mapping thread [24] to device [0], out of [1] devices...
2018-01-01T15:40:31.032 [main] DEBUG org.reflections.Reflections - going to scan these urls:
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-ppc64le.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-model/0.9.1/nd4j-parameter-server-model-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-base64/0.9.1/nd4j-base64-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-native-api/0.9.1/nd4j-native-api-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/jackson/0.9.1/jackson-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-macosx-x86_64.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson-reflectionloader/0.9.1/nd4j-jackson-reflectionloader-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-aeron/0.9.1/nd4j-aeron-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server/0.9.1/nd4j-parameter-server-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson/0.9.1/nd4j-jackson-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-api/0.9.1/nd4j-api-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-buffer/0.9.1/nd4j-buffer-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-client/0.9.1/nd4j-parameter-server-client-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-context/0.9.1/nd4j-context-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-common/0.9.1/nd4j-common-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-x86_64.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-windows-x86_64.jar!/
2018-01-01T15:40:31.144 [main] INFO  org.reflections.Reflections - Reflections took 107 ms to scan 18 urls, producing 31 keys and 227 values
2018-01-01T15:40:31.215 [main] INFO  o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CUDA]; OS: [Linux]
2018-01-01T15:40:31.216 [main] INFO  o.n.l.a.o.e.DefaultOpExecutioner - Cores: [4]; Memory: [6.0GB];
2018-01-01T15:40:31.216 [main] INFO  o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [CUBLAS]
2018-01-01T15:40:31.216 [main] INFO  o.n.l.j.o.e.CudaExecutioner - Device name: [GeForce GTX 1080 Ti]; CC: [6.1]; Total/free memory: [11711807488]
2018-01-01T15:40:31.283 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 3
2018-01-01T15:40:31.342 [main] INFO  DL4JTest1 - Build model....
2018-01-01T15:40:31.537 [main] WARN  org.reflections.Reflections - given scan urls are empty. set urls in the configuration
2018-01-01T15:40:31.656 [main] INFO  o.d.nn.multilayer.MultiLayerNetwork - Starting MultiLayerNetwork with WorkspaceModes set to [training: SEPARATE; inference: SEPARATE]
2018-01-01T15:40:31.702 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 0
2018-01-01T15:40:31.707 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 4
2018-01-01T15:40:31.752 [main] DEBUG org.reflections.Reflections - going to scan these urls:
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-ppc64le.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-model/0.9.1/nd4j-parameter-server-model-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-base64/0.9.1/nd4j-base64-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-native-api/0.9.1/nd4j-native-api-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/jackson/0.9.1/jackson-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-macosx-x86_64.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson-reflectionloader/0.9.1/nd4j-jackson-reflectionloader-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-aeron/0.9.1/nd4j-aeron-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server/0.9.1/nd4j-parameter-server-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-jackson/0.9.1/nd4j-jackson-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-api/0.9.1/nd4j-api-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-buffer/0.9.1/nd4j-buffer-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-parameter-server-client/0.9.1/nd4j-parameter-server-client-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-context/0.9.1/nd4j-context-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-common/0.9.1/nd4j-common-0.9.1.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-linux-x86_64.jar!/
jar:file:/home/fuyang/.m2/repository/org/nd4j/nd4j-cuda-8.0/0.9.1/nd4j-cuda-8.0-0.9.1-windows-x86_64.jar!/
2018-01-01T15:40:32.139 [main] INFO  org.reflections.Reflections - Reflections took 387 ms to scan 18 urls, producing 416 keys and 1637 values
2018-01-01T15:40:32.156 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 5
2018-01-01T15:40:32.195 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 2
2018-01-01T15:40:32.196 [main] DEBUG o.n.j.handler.impl.CudaZeroHandler - Creating bucketID: 1
2018-01-01T15:40:32.375 [main] INFO  org.nd4j.nativeblas.Nd4jBlas - Number of threads used for BLAS: 0
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:5057 code=77(<unknown>) "cudaStreamSynchronize(*stream)"
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4749 code=77(<unknown>) "result"
Exception in thread "main" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [imax]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.register(cudaEvent_t.java:63)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.registerAction(SynchronousFlowController.java:250)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.registerAction(CudaZeroHandler.java:1258)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.registerAction(AtomicAllocator.java:1017)
	at org.nd4j.linalg.jcublas.JCublasNDArrayFactory.concat(JCublasNDArrayFactory.java:680)
	at org.nd4j.linalg.factory.Nd4j.concat(Nd4j.java:5712)
	at org.nd4j.linalg.factory.BaseNDArrayFactory.vstack(BaseNDArrayFactory.java:1222)
	at org.nd4j.linalg.factory.Nd4j.vstack(Nd4j.java:5517)
	at DL4JTest1.main(DL4JTest1.java:107)
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
Exception in thread "UniGC thread 3" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.purgeDeviceObject(CudaZeroHandler.java:1113)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.purgeDeviceObject(AtomicAllocator.java:515)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:714)
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
Exception in thread "UniGC thread 1" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.purgeDeviceObject(CudaZeroHandler.java:1113)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.purgeDeviceObject(AtomicAllocator.java:515)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:714)
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
Exception in thread "UniGC thread 4" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.purgeDeviceObject(CudaZeroHandler.java:1113)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.purgeDeviceObject(AtomicAllocator.java:515)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:714)
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
Exception in thread "UniGC thread 2" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:696)
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
CUDA error at /home/jenkins/workspace/dl4j/all-multiplatform@2_linux-x86_64/stream1/libnd4j/blas/cuda/NativeOps.cu:4895 code=77(<unknown>) "result"
Exception in thread "UniGC thread 0" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.purgeDeviceObject(CudaZeroHandler.java:1113)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.purgeDeviceObject(AtomicAllocator.java:515)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:714)
Exception in thread "UniGC thread 5" org.nd4j.linalg.exception.ND4JException: CUDA exception happened. Terminating. Last op: [null]
	at org.nd4j.jita.allocator.pointers.cuda.cudaEvent_t.synchronize(cudaEvent_t.java:55)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillFinished(SynchronousFlowController.java:106)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillFinished(GridFlowController.java:47)
	at org.nd4j.jita.flow.impl.SynchronousFlowController.waitTillReleased(SynchronousFlowController.java:203)
	at org.nd4j.jita.flow.impl.GridFlowController.waitTillReleased(GridFlowController.java:62)
	at org.nd4j.jita.handler.impl.CudaZeroHandler.purgeDeviceObject(CudaZeroHandler.java:1113)
	at org.nd4j.jita.allocator.impl.AtomicAllocator.purgeDeviceObject(AtomicAllocator.java:515)
	at org.nd4j.jita.allocator.impl.AtomicAllocator$UnifiedGarbageCollectorThread.run(AtomicAllocator.java:714)

Process finished with exit code 1

```
