import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.iterator.MovingWindowBaseDataSetIterator;
import org.deeplearning4j.examples.recurrent.seq2seq.Seq2SeqPredicter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.Scanner;
import java.io.FileNotFoundException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.deeplearning4j.datasets.iterator.impl.MovingWindowDataSetFetcher;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ArabTrans{
	public static final int nodeCount = 128;
	public static final int vecSize = 14;
	public static int epochCount = 1;
	public static int iterationCount = 1;

	public static void main(String[] args) throws Exception{
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.INT);
        double[][] engData;
        double[][] araData;
        int count = 0;
        try{
			File file = new File("EngProcessedData.txt");
			Scanner scan = new Scanner(file);
			count = scan.nextInt();
			engData = new double[1][count];
			for (int i = 0; i < count; i++){
				engData[0][i] = scan.nextInt();
			}
			file = new File("AraProcessedData.txt");
			scan = new Scanner(file);
			count = scan.nextInt();
			araData = new double[1][count];
			for (int i = 0; i < count; i++){
				araData[0][i] = scan.nextInt();
			}
		} catch (FileNotFoundException e) {
            System.out.println("File Not Found Exception");
            engData = new double[1][1];
            araData = new double[1][1];
		}
		NDArray engNDArray = new NDArray(engData);
		NDArray araNDArray = new NDArray(araData);
		DataSet dataSet = new DataSet(engNDArray, araNDArray);
		MovingWindowDataSetFetcher fetcher = new MovingWindowDataSetFetcher(dataSet,2,count);

		ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
		.weightInit(WeightInit.XAVIER)
		.learningRate(0.25)
		.updater(Updater.ADAM)
		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(iterationCount)
		.seed(1)
		.graphBuilder()
		.addInputs("additionIn", "sumOut")
		.setInputTypes(InputType.recurrent(vecSize), InputType.recurrent(vecSize))
		.addLayer("encoder", new GravesLSTM.Builder().nIn(vecSize).nOut(nodeCount).activation(Activation.SOFTSIGN).build(),"additionIn")
		.addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
		.addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
		.addLayer("decoder", new GravesLSTM.Builder().nIn(vecSize+nodeCount).nOut(nodeCount).activation(Activation.SOFTSIGN).build(), "sumOut","duplicateTimeStep")
		.addLayer("output", new RnnOutputLayer.Builder().nIn(nodeCount).nOut(vecSize).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
		.setOutputs("output")
		.pretrain(false).backprop(true)
		.build();

		ComputationGraph net = new ComputationGraph(configuration);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		Seq2SeqPredicter predictor = new Seq2SeqPredicter(net);

		for(int i = 0; i < epochCount; i++){
			net.fit(fetcher.next());
		}
	}
}
