package de.datexis.tagger;

import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 *
 * @author sarnold
 */
public class INDArrayTest {
  
  @Test
  public void testMiniBatch() {
  
    int num = 2; // sentences
    int examplesSize = 7; //words
    int labelSize = 3; // classes
    
    INDArray label = Nd4j.zeros(num, labelSize, examplesSize);
    INDArray mask =  Nd4j.zeros(num, examplesSize);

    assertArrayEquals(new long[]{num, labelSize, examplesSize}, label.shape());
    assertArrayEquals(new long[]{num, examplesSize}, mask.shape());
    
    DataSet d0 = new DataSet(label, label, mask, mask);
    INDArray example = d0.get(0).getFeatures().get(new INDArrayIndex[] {point(0), all(), point(0)});
    assertEquals(Nd4j.create(new double[]{0,0,0}), example);
    
    for(int batchNum=0; batchNum<num; batchNum++ ) {
      for(int exampleNum=0; exampleNum<examplesSize; exampleNum++) {
        mask.put(batchNum, exampleNum, 1); // mark this word as used
        INDArray labels = Nd4j.create(new double[]{ 100*batchNum + 10*exampleNum, 100*batchNum + 10*exampleNum + 1, 100*batchNum + 10*exampleNum + 2 });
        label.vectorAlongDimension((batchNum * examplesSize) + exampleNum,1).putRow(0, labels); // label to predict
        //d0.get(batchNum).getFeatures().getColumn(exampleNum).assign(labels); // alternate option
        d0.getFeatures().getRow(batchNum).getColumn(exampleNum).assign(labels); // alternate option
			}
		}
    
    assertArrayEquals(new long[]{num, labelSize, examplesSize}, label.shape());
    assertArrayEquals(new long[]{num, examplesSize}, mask.shape());
    
    DataSet d1 = new DataSet(label, label, mask, mask);
    
    assertArrayEquals(new long[]{1, labelSize, examplesSize}, d1.get(0).getFeatures().shape());
    example = d1.get(0).getFeatures().get(new INDArrayIndex[] {point(0), all(), point(0)});
    assertEquals(Nd4j.create(new double[]{0,1,2}), example);
    example = d1.get(1).getFeatures().get(new INDArrayIndex[] {point(0), all(), point(0)});
    assertEquals(Nd4j.create(new double[]{100,101,102}), example);
    
    assertArrayEquals(new long[]{1, labelSize, examplesSize}, d0.get(0).getFeatures().shape());
    example = d0.get(0).getFeatures().get(new INDArrayIndex[] {point(0), all(), point(0)});
    assertEquals(Nd4j.create(new double[]{0,1,2}), example);
    example = d0.get(1).getFeatures().get(new INDArrayIndex[] {point(0), all(), point(0)});
    assertEquals(Nd4j.create(new double[]{100,101,102}), example);
  
  }
 
  @Test
  public void testINDArraySerialization() {
    try {
      INDArray arr = Nd4j.rand(10,1);
      Resource temp = Resource.createTempFile("indarray");
      BufferedOutputStream bos = new BufferedOutputStream(temp.getOutputStream());
      DataOutputStream dos = new DataOutputStream(bos);
      Nd4j.write(arr, dos);
      dos.flush();
      dos.close();
      
      BufferedInputStream bis = new BufferedInputStream(temp.getInputStream());
      DataInputStream dis = new DataInputStream(bis);
      INDArray ret = Nd4j.read(dis);
      dis.close();
      
      System.out.println(arr.transpose());
      System.out.println(ret.transpose());
      
      assertEquals(arr, ret);
      
    } catch (IOException ex) {
      ex.printStackTrace();
      fail();
    }
  }
  
  @Test
  public void testINDArrayBase64Serialization() {
    
    INDArray arr = Nd4j.rand(20,1);
    String encoded = ObjectSerializer.getArrayAsBase64String(arr);
    System.out.println(arr);
    System.out.println("encoded: " + encoded);
    assertNotNull(encoded);

    INDArray ret = ObjectSerializer.getArrayFromBase64String(encoded);
    System.out.println(ret);
    assertEquals(arr, ret);
    
  }
  
}
