import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.omg.PortableInterceptor.INACTIVE;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DL4JTest1 {
    private static Logger log = LoggerFactory.getLogger(DL4JTest1.class);

    private static INDArray getFeatures(int nRows, int nCols) {
        return Nd4j.rand(nRows, nCols);
    }

    private static INDArray getLabels(int nRows, int nCols) {
        INDArray labels = Nd4j.zeros(nRows, nCols);

        for (int n = 0; n < nRows; n++) {
            labels.putScalar(n, 0, 1);
        }
        return labels;
    }

    public static void main(String[] args) throws Exception {

        int numClasses = 4;     //24501 classes (types of senders) in the data set. Classes have integer values 0, 1 or 2 ... and so on
        int printIterationsNum = 200; // print score every 200 iterations
        int batchSize = 32;

        final int numInputs = 15484;
        int hiddenLayer1Num = 2000;
        int iterations = 1;
        long seed = 42;
        int nEpochs = 20;

        Nd4j.getRandom().setSeed(42);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .iterations(iterations)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
//                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hiddenLayer1Num)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(hiddenLayer1Num).nOut(numClasses).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(printIterationsNum));

        for ( int n = 0; n < nEpochs; n++) {

            //evaluate the model on the test set
            Evaluation eval = new Evaluation(numClasses);

            INDArray features = getFeatures(batchSize, numInputs);
            INDArray labels = getLabels(batchSize, numClasses);
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);

            INDArray labels_s = Nd4j.argMax(labels, 1);
            INDArray predicted_s = Nd4j.argMax(predicted, 1);

            int i = 1;
            while(true) {
                features = getFeatures(batchSize, numInputs);;
                labels = getLabels(batchSize, numClasses);
                predicted = model.output(features, false);
                eval.eval(labels, predicted);

                INDArray labels2_s = Nd4j.argMax(labels, 1);
                INDArray predicted2_s = Nd4j.argMax(predicted, 1);

                labels_s = Nd4j.vstack(labels_s,labels2_s);
                predicted_s = Nd4j.vstack(predicted_s,predicted2_s);

//                System.out.println(String.format("Epoch %d, batch number %d, labels_s shape[0] %d", n, i + 1, labels_s.shape()[0]));
                if (labels_s.shape()[0] > 20000) break; // When number of testing data is too high, we have out of memory issue on GPU.
                // TODO A CUDA error occurs when labels' lengths is above 15000
                // TODO Test this issue without GPU and make a sample code example later to see if it repeat

                i++;
            }

            // One way to evaluate accuracy
            float acc = labels_s.eq(predicted_s).mean(0).getFloat(0, 0);
            // Another way to evaluate accuracy
            log.info(String.format("Evaluation on test data - [Epoch %d] [AccHand: %.3f, Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                    n + 1, acc, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));

        }

        System.out.println("Finished...");
    }
}
