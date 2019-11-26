package de.datexis.encoder;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EncodingHelpers {
  
  public static INDArray createTimeStepMatrix(long batchSize, long vectorSize, long timeSteps) {
    return Nd4j.zeros(DataType.FLOAT, batchSize, vectorSize, timeSteps);
  }
  
  /**
   * Put a single example column vector int o a time step matrix
   * @param matrix Full batch matrix [ batch size X vector size X time steps ]
   * @param batchIndex Index of the batch
   * @param t Index of the time step
   * @param value The vector to put into matrix [ vector size X 1 ]
   */
  public static void putTimeStep(INDArray matrix, long batchIndex, long t, INDArray value) {
    //matrix.getRow(batchIndex).getColumn(t).assign(vec); // invalid operation since beta4
    //matrix.put(point(batchIndex), all(), point(t), vec); // valid operation, but had errors in beta4
    //matrix.get(point(batchIndex), all(), point(t)).assign(vec); // valid operation in beta4
    matrix.slice(batchIndex, 0).slice(t, 1).assign(value); // 25% faster
  }
  
  /**
   * Get a single example column vector from a time step matrix
   * @param matrix Full batch matrix [ batch size X vector size X time steps ]
   * @param batchIndex Index of the batch
   * @param t Index of the time step
   * @return The value as column vector [ vector size X 1 ]
   */
  public static INDArray getTimeStep(INDArray matrix, long batchIndex, long t) {
    //INDArray vec = matrix.get(point(batchIndex), all(), point(t)); // valid operation in beta4
    INDArray vec = matrix.slice(batchIndex, 0).slice(t, 1); // 25% faster
    //return vec.transpose(); // invalid operation since beta4
    return vec.reshape(matrix.size(1), 1);
  }
  
}
