import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
//import java.util.concurrent.*;

public class NeuralNet {
	private static final int TRAINING_NUM = 180; // total of 320 = 160happy 160unhappy
	private static final int TESTING_NUM = 20;
	private static final int NUM_PIXELS = 36 * 26;
	private static final String HAPPY_TRAIN_PATH = "./happyTraining.csv";
	private static final String UNHAPPY_TRAIN_PATH = "./unhappyTraining.csv";
	private static final String HAPPY_TEST_PATH = "./happyTest.csv";
	private static final String UNHAPPY_TEST_PATH = "./unhappyTest.csv";
	private static final String OUTPUT_TEXT_PATH = "./output.txt";
	private static final String OUTPUT_CSV_PATH = "./activations.csv";
	private static final int MAX_EPOCHS = 30;
	private static final Double COST_FINAL_MAX = 35.0;
	private static final Double LEARNING_RATE = 0.02;

	public static void main(String[] args) {
		System.out.println("reading training/testing data");
		try {
			// read face data
			Faces faceData = new Faces(TRAINING_NUM, TESTING_NUM, NUM_PIXELS, HAPPY_TRAIN_PATH, UNHAPPY_TRAIN_PATH,
					HAPPY_TEST_PATH, UNHAPPY_TEST_PATH);
			System.out.println("Done Reading");

			// initialize weights and biases
			Weights w = initializeWeights();
			Biases b = initializeBiases();
			// and activations for current epoch
			Pair<Double[], Double> ai = new Pair<Double[], Double>(new Double[NUM_PIXELS], 0.0);
			// and cost
			Double cCurr = 0.0;
			Double cPrev = 999.9;
			Double eps = 999.9;
			// and gradients
			Gradients gradients = new Gradients(NUM_PIXELS);

			// calculate cost
			cCurr = 0.0;
			for (int ii = 0; ii < 2 * TRAINING_NUM; ii++) {
				calculateActivations(ai, faceData.xTraining[ii], w, b);
				cCurr += Math.pow(faceData.yTraining[ii] - ai.second(), 2);
			}
			cCurr *= 0.5;
			System.out.println("initial cost: " + cCurr);

			for (int i = 1; i <= MAX_EPOCHS && cCurr > COST_FINAL_MAX; i++) {
				System.out.println("epoch number " + i);
				// shuffle training data
				faceData.shuffleTraining();
				System.out.println("Shuffled training data");

				cPrev = cCurr;
				cCurr = 0.0; // will accumulate cost over course of epoch

				for (int ii = 0; ii < 2 * TRAINING_NUM; ii++) {
					System.out.print(".");
//					// calculate activations
					calculateActivations(ai, faceData.xTraining[ii], w, b);
					// add layer 2 activation error to cost
					cCurr += Math.pow(faceData.yTraining[ii] - ai.second(), 2);
					// calculate gradients
					calculateGradients(gradients, faceData.xTraining[ii], faceData.yTraining[ii], ai.first(),
							ai.second(), w, b);
					// update weights
					// layer1
					for (int j = 0; j < NUM_PIXELS; j++)
						for (int jj = 0; jj < NUM_PIXELS; jj++)
							w.first()[jj][j] -= LEARNING_RATE * gradients.dcdw().first()[jj][j];
					// layer2
					for (int j = 0; j < NUM_PIXELS; j++)
						w.second()[j] -= LEARNING_RATE * gradients.dcdw().second()[j];
					// update biases
					// layer1
					for (int j = 0; j < NUM_PIXELS; j++)
						b.first()[j] -= LEARNING_RATE * gradients.dcdb().first()[j];
					// layer 2
					b.setSecond(b.second() - LEARNING_RATE * gradients.dcdb().second());
				}

				cCurr *= 0.5; // finally, divide by 2
				System.out.println("\nCurrent cost: " + cCurr);

				eps = Math.abs(cCurr - cPrev);
				System.out.println("Current epsilon: " + eps);
			} // end epoch
			Double[] hidden = new Double[NUM_PIXELS];
			// Make predictions
			Double[] yHat = new Double[2 * TESTING_NUM];
			Double numCorrect = 0.0;
			for (int i = 0; i < 2 * TESTING_NUM; i++) {
				calculateActivations(ai, faceData.xTesting[i], w, b);
				if (i == 0)
					hidden = ai.first();
				if (ai.second() >= 0.5 && faceData.yTesting[i] == 1.0) {
					numCorrect += 1.0;
					yHat[i] = 1.0;
					System.out.println("Predicted Happy, was happy. a = " + ai.second());
				} else if (ai.second() < 0.5 && faceData.yTesting[i] == 0.0) {
					numCorrect += 1.0;
					yHat[i] = 0.0;
					System.out.println("Predicted Unhappy, wasn't happy. a = " + ai.second());
				} else if (ai.second() >= 0.5 && faceData.yTesting[i] == 0.0) {
					yHat[i] = 1.0;
					System.out.println("Predicted Happy, wasn't happy. a = " + ai.second());
				} else if (faceData.yTesting[i] == 1.0) {
					yHat[i] = 0.0;
					System.out.println("Prediced Unhappy, was happy. a = " + ai.second());
				} else
					System.out.println("error: test data y value not 1.0 or 0.0");
			}

			System.out.println(
					"" + numCorrect + " of " + (2 * TESTING_NUM) + " correct. " + (numCorrect / (2 * TESTING_NUM)));

			// write output.txt and .csv file of final activations
			writePredictions(yHat);
			writeCSV(hidden, OUTPUT_CSV_PATH);
			writeCSV(w.second(), "./hiddenweights.csv");

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static Weights initializeWeights() {
		Random rand = new Random();
		Double[][] w1 = new Double[NUM_PIXELS][NUM_PIXELS];
		Double[] w2 = new Double[NUM_PIXELS];
		for (int i = 0; i < NUM_PIXELS; i++) {
			w2[i] = rand.nextDouble() - rand.nextDouble();
			for (int j = 0; j < NUM_PIXELS; j++)
				w1[i][j] = rand.nextDouble() - rand.nextDouble();
		}
		return new Weights(w1, w2);
	}

	public static Biases initializeBiases() {
		Random rand = new Random();
		Double[] b1 = new Double[NUM_PIXELS];
		for (int i = 0; i < NUM_PIXELS; i++)
			b1[i] = rand.nextDouble() - rand.nextDouble();
		Double b2 = rand.nextDouble() - rand.nextDouble();
		return new Biases(b1, b2);
	}

	public static void calculateActivations(Pair<Double[], Double> a, Double[] x, Weights w, Biases b) {
		// layer 1
		for (int j = 0; j < NUM_PIXELS; j++) {
			Double xw = 0.0;
			for (int jj = 0; jj < NUM_PIXELS; jj++)
				xw += x[jj] * w.first()[jj][j];
			a.first()[j] = 1 / (1 + Math.exp(-1 * (xw + b.first()[j])));
		}
		// layer 2
		Double aw = 0.0;
		for (int j = 0; j < NUM_PIXELS; j++)
			aw += a.first()[j] * w.second()[j];
		a.setSecond(1 / (1 + Math.exp(-1 * (aw + b.second()))));
	}

	// calculate gradients. Returns pair of pairs. First pair is dc/dw in layer 1
	// and layer 2 respectively
	// Second pair contains dc/db for layer 1 and layer 2 respectively
	public static void calculateGradients(Gradients gradients, Double[] xi, Double yi, Double[] ai1, Double ai2,
			Weights w, Biases b) {
		gradients.dcdb().setSecond((ai2 - yi) * ai2 * (1 - ai2));
		for (int j = 0; j < NUM_PIXELS; j++) {
			gradients.dcdw().second()[j] = gradients.dcdb().second() * ai1[j];
			gradients.dcdb().first()[j] = gradients.dcdw().second()[j] * w.second()[j] * (1 - ai1[j]);
			for (int jj = 0; jj < NUM_PIXELS; jj++)
				gradients.dcdw().first()[jj][j] = gradients.dcdb().first()[j] * xi[jj];
		}
	}

	public static void writePredictions(Double[] yHat) throws FileNotFoundException, IOException {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_TEXT_PATH))) {
			for (int i = 0; i < yHat.length; i++) {
				bw.write("" + Math.round(yHat[i]));
				bw.newLine();
			}
		}
	}

	public static void writeCSV(Double[] activations, String path) throws FileNotFoundException, IOException {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
			bw.write("" + Double.toString(activations[0]));
			for (int i = 1; i < activations.length; i++)
				bw.write("," + Double.toString(activations[i]));
		}
	}
}

class Pair<T, V> {
	T t;
	V v;

	Pair(T t, V v) {
		this.t = t;
		this.v = v;
	}

	T first() {
		return t;
	}

	V second() {
		return v;
	}

	void setFirst(T t) {
		this.t = t;
	}

	void setSecond(V v) {
		this.v = v;
	}

	Boolean equals(Pair<T, V> other) {
		return (t == other.first() && v == other.second());
	}
}

class Faces {
	Double[] yTraining;
	Double[] yTesting;
	Double[][] xTraining;
	Double[][] xTesting;

	Faces(int TRAINING_NUM, int TESTING_NUM, int NUM_PIXELS, String happyTrainPath, String unhappyTrainPath,
			String happyTestPath, String unhappyTestPath) throws FileNotFoundException, IOException {
		Pair<Pair<Double[], Double[][]>, Pair<Double[], Double[][]>> facesData = readFaceData(happyTrainPath,
				unhappyTrainPath, happyTestPath, unhappyTestPath, TRAINING_NUM, TESTING_NUM, NUM_PIXELS);
		this.yTraining = facesData.first().first();
		this.yTesting = facesData.second().first();
		this.xTraining = facesData.first().second();
		this.xTesting = facesData.second().second();
	}

	// reads training and test data. Returns pair of training, test. Data is pair of
	// y, x.
	public static Pair<Pair<Double[], Double[][]>, Pair<Double[], Double[][]>> readFaceData(String happyTrainPath,
			String unhappyTrainPath, String happyTestPath, String unhappyTestPath, int TRAINING_NUM, int TESTING_NUM,
			int NUM_PIXELS) throws FileNotFoundException, IOException {
		Double[] yTraining = new Double[2 * TRAINING_NUM];
		Double[][] xTraining = new Double[2 * TRAINING_NUM][NUM_PIXELS];
		Double[] yTesting = new Double[2 * TESTING_NUM];
		Double[][] xTesting = new Double[2 * TESTING_NUM][NUM_PIXELS];
		// read training files
		try (BufferedReader brHappy = new BufferedReader(new FileReader(happyTrainPath));
				BufferedReader brUnhappy = new BufferedReader(new FileReader(unhappyTrainPath))) {
			for (int i = 0; i < TRAINING_NUM; i++) {
				yTraining[i] = 1.0;
				yTraining[i + TRAINING_NUM] = 0.0;
			}
			String nxtLine;
			// parse happy
			for (int j = 0; ((nxtLine = brHappy.readLine()) != null && j < TRAINING_NUM); j++)
				for (int k = 0; k < NUM_PIXELS; k++)
					xTraining[j][k] = Double.parseDouble(nxtLine.split(",")[k])/ 255.0;
			// parse unhappy
			for (int j = 0; ((nxtLine = brUnhappy.readLine()) != null && j < TRAINING_NUM); j++)
				for (int k = 0; k < NUM_PIXELS; k++)
					xTraining[j + TRAINING_NUM][k] = Double.parseDouble(nxtLine.split(",")[k]) / 255.0;
		}
		// read testing files
		try (BufferedReader brHappy = new BufferedReader(new FileReader(happyTestPath));
				BufferedReader brUnhappy = new BufferedReader(new FileReader(unhappyTestPath))) {
			for (int i = 0; i < TESTING_NUM; i++) {
				yTesting[2 * i] = 1.0;
				yTesting[2 * i + 1] = 0.0;
			}
			// parse happy
			String nxtLine;
			for (int j = 0; ((nxtLine = brHappy.readLine()) != null && j < TESTING_NUM); j++)
				for (int k = 0; k < NUM_PIXELS; k++)
					xTesting[j * 2][k] = Double.parseDouble(nxtLine.split(",")[k]) / 255.0;
			// parse unhappy
			for (int j = 0; ((nxtLine = brUnhappy.readLine()) != null && j < TESTING_NUM); j++)
				for (int k = 0; k < NUM_PIXELS; k++)
					xTesting[j * 2 + 1][k] = Double.parseDouble(nxtLine.split(",")[k]) / 255.0;
		}

		Pair<Double[], Double[][]> trainingFaces = new Pair<Double[], Double[][]>(yTraining, xTraining);
		Pair<Double[], Double[][]> testingFaces = new Pair<Double[], Double[][]>(yTesting, xTesting);
		return new Pair<Pair<Double[], Double[][]>, Pair<Double[], Double[][]>>(trainingFaces, testingFaces);
	}

	// shuffle the training data. Assume lengths are correct
	public void shuffleTraining() {
		Random rand = new Random();
		int len = yTraining.length;
		for (int i = len - 1; i > 0; i--) {
			int randInt = rand.nextInt(i);
			Double tmp1 = yTraining[i];
			Double[] tmp2 = xTraining[i];
			yTraining[i] = yTraining[randInt];
			xTraining[i] = xTraining[randInt];
			yTraining[randInt] = tmp1;
			xTraining[randInt] = tmp2;
		}
	}
}

class Weights {
	private Pair<Double[][], Double[]> w;

	Weights(Double[][] w1, Double[] w2) {
		this.w = new Pair<Double[][], Double[]>(w1, w2);
	}

	Double[][] first() {
		return this.w.first();
	}

	Double[] second() {
		return this.w.second();
	}

	void setFirst(Double[][] w1) {
		this.w.setFirst(w1);
	}

	void setSecond(Double[] w2) {
		this.w.setSecond(w2);
		;
	}
}

class Biases {
	private Pair<Double[], Double> b;

	Biases(Double[] b1, Double b2) {
		this.b = new Pair<Double[], Double>(b1, b2);
	}

	Double[] first() {
		return this.b.first();
	}

	Double second() {
		return this.b.second();
	}

	void setFirst(Double[] b1) {
		this.b.setFirst(b1);
	}

	void setSecond(Double b2) {
		this.b.setSecond(b2);
		;
	}
}

class Gradients {
	Weights dcdw;
	Biases dcdb;

	Gradients(int numPixels) {
		this.dcdw = new Weights(new Double[numPixels][numPixels], new Double[numPixels]);
		this.dcdb = new Biases(new Double[numPixels], 0.0);
	}

	Gradients(Weights dcdw, Biases dcdb) {
		this.dcdw = dcdw;
		this.dcdb = dcdb;
	}

	Weights dcdw() {
		return this.dcdw;
	}

	Biases dcdb() {
		return this.dcdb;
	}

	void set_dcdw(Weights dcdw) {
		this.dcdw = dcdw;
	}

	void set_dcdb(Biases dcdb) {
		this.dcdb = dcdb;
	}
}