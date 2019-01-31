package de.datexis.sector.eval;

import java.io.IOException;
import org.deeplearning4j.eval.EvaluationUtils;
import org.deeplearning4j.util.TimeSeriesUtils;
import static org.junit.Assert.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorEvaluationTest {

  protected final static Logger log = LoggerFactory.getLogger(SectorEvaluationTest.class);
  
  @Test
  public void testDatasetReshaping() throws IOException {
    
    System.out.println("de.datexis.models.sector.eval.SectorEvaluationTest.testDatasetReshaping()");
    
    int batchSize = 3;
    int outputSize = 2;
    int maxTimeSteps = 5;
    
    // shape [miniBatchSize,outputSize,timeSeriesLength] 
    INDArray labels = Nd4j.zeros(batchSize, outputSize, maxTimeSteps);
    // shape [miniBatchSize,timeSeriesLength]
    INDArray outputMask = Nd4j.zeros(batchSize, maxTimeSteps, 'f');
    // shape [examples,outputSize]
    INDArray labels2D = Nd4j.zeros(10, outputSize);

    // Document 1: [ 1 1 ] [ 1 0 ] [ 0 1 ]
    for(int t = 0; t < 3; t++) outputMask.putScalar(new int[] {0, t}, 1);
    labels.put(new INDArrayIndex[] {point(0), all(), point(0)}, Nd4j.create(new double[]{ .1, .1 }));
    labels.put(new INDArrayIndex[] {point(0), all(), point(1)}, Nd4j.create(new double[]{ .2, .0 }));
    labels.put(new INDArrayIndex[] {point(0), all(), point(2)}, Nd4j.create(new double[]{ .0, .3 }));
    
    // Document 2: [ 1 0 ] [ 1 1 ]
    for(int t = 0; t < 2; t++) outputMask.putScalar(new int[] {1, t}, 1);
    labels.put(new INDArrayIndex[] {point(1), all(), point(0)}, Nd4j.create(new double[]{ .4, .0 }));
    labels.put(new INDArrayIndex[] {point(1), all(), point(1)}, Nd4j.create(new double[]{ .5, .5 }));
    
    // Document 3: [ 0 1 ] [ 0 0 ] [ 1 1 ] [ 1 1 ] [ 1 0 ]
    for(int t = 0; t < 5; t++) outputMask.putScalar(new int[] {2, t}, 1);
    labels.put(new INDArrayIndex[] {point(2), all(), point(0)}, Nd4j.create(new double[]{ .0, .6 }));
    labels.put(new INDArrayIndex[] {point(2), all(), point(1)}, Nd4j.create(new double[]{ .0, .0 }));
    labels.put(new INDArrayIndex[] {point(2), all(), point(2)}, Nd4j.create(new double[]{ .7, .7 }));
    labels.put(new INDArrayIndex[] {point(2), all(), point(3)}, Nd4j.create(new double[]{ .8, .8 }));
    labels.put(new INDArrayIndex[] {point(2), all(), point(4)}, Nd4j.create(new double[]{ .9, .0 }));
    
    // All documents together in one 2d batch
    labels2D.getRow(0).assign(Nd4j.create(new double[]{ .1, .1 }));
    labels2D.getRow(1).assign(Nd4j.create(new double[]{ .2, .0 }));
    labels2D.getRow(2).assign(Nd4j.create(new double[]{ .0, .3 }));
    labels2D.getRow(3).assign(Nd4j.create(new double[]{ .4, .0 }));
    labels2D.getRow(4).assign(Nd4j.create(new double[]{ .5, .5 }));
    labels2D.getRow(5).assign(Nd4j.create(new double[]{ .0, .6 }));
    labels2D.getRow(6).assign(Nd4j.create(new double[]{ .0, .0 }));
    labels2D.getRow(7).assign(Nd4j.create(new double[]{ .7, .7 }));
    labels2D.getRow(8).assign(Nd4j.create(new double[]{ .8, .8 }));
    labels2D.getRow(9).assign(Nd4j.create(new double[]{ .9, .0 }));
    
    assertEquals(3, labels.rank());
    assertArrayEquals(new long[] { batchSize, outputSize, maxTimeSteps }, labels.shape());
    assertEquals(2, outputMask.rank());
    assertArrayEquals(new long[] { batchSize, maxTimeSteps }, outputMask.shape());
    assertEquals(2, labels2D.rank());
    assertArrayEquals(new long[] { 3 + 2 + 5, outputSize }, labels2D.shape());

    //Reshaping here: basically RnnToFeedForwardPreProcessor...
    //Dup to f order, to ensure consistent buffer for reshaping
    labels = labels.dup('f');
    
    // This function should normally work:
    Pair<INDArray, INDArray> test = EvaluationUtils.extractNonMaskedTimeSteps(labels, labels, outputMask);
    assertEquals(2, test.getFirst().rank());
    assertArrayEquals(new long[] { 10, outputSize }, test.getFirst().shape());
    assertEquals(labels2D.sumNumber().doubleValue(), test.getFirst().sumNumber().doubleValue(), 1e-5);
    // order is not maintained!
    //assertEquals(labels2D, test.getFirst());

    /*INDArray labelsReshaped = EvaluationUtils.reshapeTimeSeriesTo2d(labels);
    assertEquals(2, labelsReshaped.rank());
    assertArrayEquals(new int[] { batchSize * maxTimeSteps, outputSize }, labelsReshaped.shape()); // still contains [ 0 0 ] from masked time steps
    
    INDArray oneDMask = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(outputMask);*/
    /*float[] f = oneDMask.dup().data().asFloat();
    int[] rowsToPull = new int[f.length];
    int usedCount = 0;
    for(int i = 0; i < f.length; i++) {
      if(f[i] == 1.0f) {
        rowsToPull[usedCount++] = i;
      }
    }
    if(usedCount == 0) {
      //Edge case: all time steps are masked -> nothing to extract
      return null;
    }
    rowsToPull = Arrays.copyOfRange(rowsToPull, 0, usedCount);

    labels2d = Nd4j.pullRows(labels2d, 1, rowsToPull);
    predicted2d = Nd4j.pullRows(predicted2d, 1, rowsToPull);

    return new Pair<>(labels2d, predicted2d);*/
    
  }

}
