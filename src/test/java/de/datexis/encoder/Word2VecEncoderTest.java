package de.datexis.encoder;

import de.datexis.common.Resource;
import de.datexis.encoder.impl.Word2VecEncoder;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Word2VecEncoderTest {
  
  private final static Resource txt = Resource.fromJAR("encoder/word2vec.txt");
  
  public Word2VecEncoderTest() {
  }
  
  @Test
  public void testLoadModel() {
    Word2VecEncoder vec = Word2VecEncoder.load(txt);
    vec.setPreprocessor(new MinimalLowercasePreprocessor());
    assertEquals(150, vec.getEmbeddingVectorSize());
    assertTrue(vec.isUnknown("DATEXIS"));
    assertFalse(vec.isUnknown("berlin"));
    assertFalse(vec.isUnknown("Berlin"));
    assertFalse(vec.isUnknown("kuestenstrasse"));
    assertFalse(vec.isUnknown("Küstenstraße"));
    assertFalse(vec.isUnknown("5-minuten-takt"));
    assertFalse(vec.isUnknown("30-Minuten-Takt"));
    assertTrue(vec.isUnknown("#-minuten-takt"));
    assertTrue(vec.isUnknown("Berlin Berlin"));
    assertTrue(vec.isUnknown("Berlin Küstenstraße"));
  }
  
  @Test
  public void testSaveBinaryModel() {
    Word2VecEncoder vec = Word2VecEncoder.load(txt);
    vec.setPreprocessor(new MinimalLowercasePreprocessor());
    assertEquals(150, vec.getEmbeddingVectorSize());
    Resource temp = Resource.createTempDirectory();
    vec.saveModel(temp, "word2vec", Word2VecEncoder.ModelType.BINARY);
    Word2VecEncoder bin = Word2VecEncoder.load(temp.resolve("word2vec.bin"));
    bin.setPreprocessor(new MinimalLowercasePreprocessor());
    assertEquals(150, bin.getEmbeddingVectorSize());
    assertTrue(bin.isUnknown("DATEXIS"));
    assertFalse(bin.isUnknown("berlin"));
    assertFalse(bin.isUnknown("Berlin"));
    assertFalse(bin.isUnknown("kuestenstrasse"));
    assertFalse(bin.isUnknown("Küstenstraße"));
    assertFalse(bin.isUnknown("5-minuten-takt"));
    assertFalse(bin.isUnknown("30-Minuten-Takt"));
    assertTrue(bin.isUnknown("#-minuten-takt"));
    assertTrue(bin.isUnknown("Berlin Berlin"));
    assertTrue(bin.isUnknown("Berlin Küstenstraße"));
    assertEquals(vec.encode("berlin"), bin.encode("berlin"));
    assertEquals(Nd4j.zeros(150, 1), vec.encode("DATEXIS")); // unknown word should give nullvector
    assertNotEquals(Nd4j.zeros(150, 1), vec.encode("berlin"));
    assertEquals(vec.encode("Berlin"), vec.encode("Berlin Berlin")); // should be mean vector
    assertNotEquals(Nd4j.zeros(150, 1), vec.encode("DATEXIS Berlin")); // should not be totally unknown
    assertNotEquals(Nd4j.zeros(150, 1), vec.encode("Berlin Küstenstraße"));
    assertNotEquals(vec.encode("Berlin"), vec.encode("Berlin Küstenstraße")); // should be something different
  }
  
  @Test
  public void testEncodings() {
    Word2VecEncoder vec = Word2VecEncoder.load(txt);
    vec.setPreprocessor(new MinimalLowercasePreprocessor());
    INDArray a = vec.encode("berlin");
    // this has to pass for all Encoders. Don't change!
    long size = vec.getEmbeddingVectorSize();
    assertEquals(size, a.length());
    assertEquals(size, a.size(0));
    assertEquals(size, a.rows());
    assertEquals(1, a.size(1));
    assertEquals(1, a.columns());
    assertEquals(2, a.rank());
  }
  
}
