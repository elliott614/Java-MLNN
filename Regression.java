import java.util.ArrayList;
import java.util.Iterator;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class Regression {
	private static final int M = 784;
	private static final String OUTPUT_FILE_PATH = "./output.txt";
	private static final String WEIGHTS_FILE_PATH = "./weights.csv";

	public static void main(String[] args) throws IOException {
		// check number of arguments
		if (args.length != 8) {
			System.out.println("Invalid number of arguments.");
			System.out.println(
					"Should be: training data path, test data path 1, test data path 2, digit 1, digit 2, learning rate, epsilon, max iterations");
			return;
		}

		// read arguments
		final String TRAINING_DATA_PATH = args[0];
		final String TEST_DATA_PATH_1 = args[1];
		final String TEST_DATA_PATH_2 = args[2];
		final String DIGIT_1 = args[3];
		final String DIGIT_2 = args[4];
		try {
			final double LEARNING_RATE = Double.parseDouble(args[5]);
			final double EPSILON = Double.parseDouble(args[6]);
			final int MAX_ITERATIONS = Integer.parseInt(args[7]);

			// parse training data (Step 2)
			System.out.println("Reading Training Data");
			Pair<List<Double>, List<Double[]>> trainingData = readTrainingData(TRAINING_DATA_PATH, DIGIT_1, DIGIT_2);
			List<Double> y = trainingData.first();
			List<Double[]> x = trainingData.second();
			System.out.println("n = " + y.size());

			// initialize w, b, current/previous costs, epsilon (beginning of Step 3)
			Random rand = new Random();
			Double[] w = new Double[M];
			for (int i = 0; i < M; i++)
				w[i] = rand.nextDouble();
			Double b = rand.nextDouble();
			Double curr_cost = 0.0;
			Double prev_cost;
			Double eps = 9999999.9; // something large to start with

			// begin iterations of steps 3/4 i.e. train weights/bias
			for (int iteration = 1; eps >= EPSILON && iteration <= MAX_ITERATIONS; iteration++) {
				System.out.println("Current iteration: " + iteration);
				// move current cost to previous cost
				prev_cost = curr_cost;

				// calculate a, update weights/bias/cost/epsilon
				Double[] a = calculate_ai(x, w, b);
				w = updateWeights(x, y, w, a, LEARNING_RATE);
				b = updateBias(y, b, a, LEARNING_RATE);
				curr_cost = calculateCost(y, a);
				eps = Math.abs(curr_cost - prev_cost);
				System.out.println("cost = " + curr_cost + ", epsilon = " + eps);
			}

			// Step 6 (TESTING!)
			List<Double[]> testData1 = new ArrayList<Double[]>();
			testData1.addAll(readTestData(TEST_DATA_PATH_1));
			List<Double[]> testData2 = new ArrayList<Double[]>();
			testData2.addAll(readTestData(TEST_DATA_PATH_2));

			List<String> predictions1 = makePrediction(testData1, w, b, DIGIT_1, DIGIT_2);
			List<String> predictions2 = makePrediction(testData2, w, b, DIGIT_1, DIGIT_2);

			double correctGuesses = 0;
			for (int i = 0; i < predictions1.size(); i++) {
				if (predictions1.get(i).equals(DIGIT_1)) {
					correctGuesses++;
					System.out.println("Prediction " + (i + 1) + " for digit " + DIGIT_1 + " is Correct");
				} else
					System.out.println("Prediction " + (i + 1) + " for digit " + DIGIT_1 + " is Incorrect");
			}
			for (int i = 0; i < predictions2.size(); i++) {
				if (predictions2.get(i).equals(DIGIT_2)) {
					correctGuesses++;
					System.out.println("Prediction " + (i + 1) + " for digit " + DIGIT_2 + " is Correct");
				} else
					System.out.println("Prediction " + (i + 1) + " for digit " + DIGIT_2 + " is Incorrect");
			}

			System.out.println("Accuracy: " + correctGuesses / (testData1.size() + testData2.size()));

			// write output file output.txt
			writeOutput(predictions1, predictions2, OUTPUT_FILE_PATH);
			
			// write output file weights.csv
			writeWeights(w, WEIGHTS_FILE_PATH);

		} catch (NumberFormatException e) {
			System.out.println(
					"Parsing error. Learning rate/epsilon should be doubles, and max iterations should be an int");
		}
	}

	// Returns a Pair whose first element is y and second element is x from
	// assignment instructions.
	public static Pair<List<Double>, List<Double[]>> readTrainingData(String path, String digit1, String digit2)
			throws FileNotFoundException, IOException {
		List<Double[]> x = new ArrayList<>(); // we do not know n as it depends on the digits
		List<Double> y = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(path));) {
			String current_line = br.readLine();
			while (current_line != null) {
				String[] vals = current_line.split(",");
				if (vals[0].contentEquals(digit1)) {
					y.add(0.0);
					Double[] tmp = new Double[M];
					for (int i = 1; i <= M; i++)
						tmp[i - 1] = Double.parseDouble(vals[i]) / 255.0;
					x.add(tmp);
				} else if (vals[0].contentEquals(digit2)) {
					y.add(1.0);
					Double[] tmp = new Double[M];
					for (int i = 1; i <= M; i++)
						tmp[i - 1] = Double.parseDouble(vals[i]) / 255.0;
					x.add(tmp);
				}
				current_line = br.readLine();
			}
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("invalid training data file '" + path + "'. Must have M columns.");
		}
		return new Pair<List<Double>, List<Double[]>>(y, x);
	}

	public static List<Double[]> readTestData(String path) throws FileNotFoundException, IOException {
		List<Double[]> result = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(path));) {
			String current_line = br.readLine();
			while (current_line != null) {
				String[] vals = current_line.split(",");
				Double[] tmp = new Double[M];
				for (int i = 0; i < M; i++)
					tmp[i] = Double.parseDouble(vals[i]) / 255.0;
				result.add(tmp);
				current_line = br.readLine();
			}
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("invalid test data file '" + path + "'. Must have M columns.");
		}
		return result;
	}

	public static Double[] calculate_ai(List<Double[]> x, Double[] w, Double b) {
		Double[] a = new Double[x.size()];
		for (int i = 0; i < x.size(); i++) {
			double sum_wjxij = 0;
			for (int j = 0; j < M; j++) {
				sum_wjxij += w[j] * x.get(i)[j];
			}
			a[i] = 1.0 / (1 + Math.exp(-1 * (sum_wjxij + b)));
		}
		return a;
	}

	public static Double[] updateWeights(List<Double[]> x, List<Double> y, Double[] w, Double[] a,
			Double learningRate) {
		for (int j = 0; j < M; j++) {
			double tmp = 0;
			for (int i = 0; i < x.size(); i++)
				tmp += (a[i] - y.get(i)) * x.get(i)[j];
			w[j] -= learningRate * tmp; // update weights
		}
		return w;
	}

	public static Double updateBias(List<Double> y, Double b, Double[] a, Double learningRate) {
		double tmp = 0;
		for (int i = 0; i < y.size(); i++)
			tmp += (a[i] - y.get(i));
		return b - learningRate * tmp;
	}

	public static Double calculateCost(List<Double> y, Double[] a) {
		Double cost = 0.0;
		for (int i = 0; i < y.size(); i++) {
			if (y.get(i) == 0.0 && a[i] > 0.9999)
				cost += 100.0;
			else if (y.get(i) == 0.0)
				cost += -1 * Math.log(1 - a[i]);
			else if (y.get(i) == 1.0 && a[i] < 0.0001)
				cost += 100.0;
			else
				cost += -1 * Math.log(a[i]);
		}
		return cost;
	}

	// Makes a prediction of either DIGIT_1 or DIGIT_2 depending on the input
	// weights and test data
	public static List<String> makePrediction(List<Double[]> testData, Double[] w, Double b, String digit1,
			String digit2) {
		List<String> predictions = new ArrayList<>();
		for (int i = 0; i < testData.size(); i++) {
			double sum_wixij = 0;
			for (int j = 0; j < M; j++)
				sum_wixij += w[j] * testData.get(i)[j];
			if (1.0 / (1 + Math.exp(-1 * (sum_wixij + b))) < 0.5)
				predictions.add(digit1);
			else
				predictions.add(digit2);
		}
		return predictions;
	}

	public static void writeOutput(List<String> predictions1, List<String> predictions2, String path)
			throws IOException {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path));) {
			Iterator<String> p1_iter = predictions1.iterator();
			Iterator<String> p2_iter = predictions2.iterator();
			String nxt;
			while (p1_iter.hasNext()) {
				nxt = p1_iter.next();
				bw.write(nxt);
				;
				bw.newLine();
			}
			while (p2_iter.hasNext()) {
				nxt = p2_iter.next();
				bw.write(nxt);
				;
				bw.newLine();
			}
		}
	}

	public static void writeWeights(Double[] w, String path) throws IOException {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path));) {
			//get max and min
			double max = 0;
			double min = 0;
			for (int i = 0; i < w.length - 1; i++) {
				if (w[i] >= max) max = w[i];
				if (w[i] <= min) min = w[i];
			}

			for (int i = 0; i < w.length - 1; i++) {
				bw.write("" + Math.round(255 * (w[i] - min) / (max - min)));
				bw.write(",");
			}
			bw.write("" +  Math.round(255 * (w[w.length - 1] - min) / (max - min)));
		}
	}
}
