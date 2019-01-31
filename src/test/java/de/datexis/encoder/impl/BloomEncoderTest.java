package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.model.Document;
import de.datexis.preprocess.DocumentFactory;
import java.util.Arrays;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class BloomEncoderTest {

  protected final static Logger log = LoggerFactory.getLogger(BloomEncoderTest.class);
  
  final private String text = "Sitta carolinensis. The white-breasted nuthatch (Sitta carolinensis) is a small that breeds in old-growth woodland across much "
      + "of temperate North America. It is a stocky nuthatch with a large head, short tail, powerful bill, and strong feet. The upperparts are pale "
      + "blue-gray, and the face and underparts are white. It has a black cap and a chestnut lower belly. The nine subspecies differ mainly in the "
      + "color of the body plumage. Like other nuthatches, the white-breasted nuthatch forages for insects on trunks and branches and is able to move "
      + "head-first down trees. Seeds form a substantial part of its winter diet, as do acorns and hickory nuts that were stored by the bird in the "
      + "fall. The nest is in a hole in a tree, and a breeding pair may smear insects around the entrance as a deterrent to squirrels. Adults and "
      + "their young may be killed by hawks, owls, and snakes, and forest clearance may lead to local habitat loss, but this is a common species with "
      + "no major conservation concerns over most of its range. ";
  
  @Test
  public void testBloomEncoder() {
    Document doc = DocumentFactory.fromText(text);
    BloomEncoder enc = new BloomEncoder();
    assertFalse(enc.isModelAvailable());
    enc.trainModel(Arrays.asList(doc), 0, WordHelpers.Language.EN);
    assertTrue(enc.isModelAvailable());
    long vectorSize = enc.getEmbeddingVectorSize();
    assertTrue(vectorSize >= 8);
    INDArray word1 = enc.encode("nuthatch");
    INDArray word2 = enc.encode("songbird");
    INDArray word3 = enc.encode("microsoft");
    assertFalse(word1.equals(word2));
    assertFalse(word2.equals(word3));
    assertFalse(word1.equals(word3));
    assertTrue(word1.sumNumber().intValue() > 0);
    assertTrue(word2.sumNumber().intValue() > 0);
    //assertTrue(word3.sumNumber().intValue() == 0);
  }
  
  @Test
  public void testBloomEncoderSentences() {
    Document doc = DocumentFactory.fromText(text);
    BloomEncoder enc = new BloomEncoder(1024, 4);
    enc.trainModel(Arrays.asList(doc), 0, WordHelpers.Language.EN);
    assertEquals(1024, enc.getEmbeddingVectorSize());
    INDArray sent1 = enc.encode(doc.getSentence(0));
    INDArray sent2 = enc.encode(doc.getSentence(1));
    INDArray word1 = enc.encode("Sitta");
    INDArray word2 = enc.encode("carolinensis");
    INDArray word3 = enc.encode(".");
    assertFalse(word1.equals(word2));
    assertFalse(word2.equals(word3));
    assertFalse(word1.equals(word3));
    assertFalse(sent1.equals(sent2));
    INDArray sent3 = word1.add(word2).add(word3);
    assertEquals(sent3, sent1);
  }
  
  @Test
  public void saveLoadBloomEncoder() {
    Document doc = DocumentFactory.fromText(text);
    Resource temp = Resource.createTempDirectory();
    
    BloomEncoder enc = new BloomEncoder(1024, 4);
    assertFalse(enc.isModelAvailable());
    enc.trainModel(Arrays.asList(doc), 0, WordHelpers.Language.EN);
    assertTrue(enc.isModelAvailable());
    INDArray sent1 = enc.encode(doc.getSentence(0));
    INDArray word1 = enc.encode("carolinensis");
    assertFalse(sent1.equals(word1));
    assertEquals(1024, enc.getEmbeddingVectorSize());
    assertEquals(1024, sent1.length());
    assertEquals(1024, word1.length());
    enc.saveModel(temp, "bloom");
    
    BloomEncoder enc2 = new BloomEncoder();
    assertFalse(enc2.isModelAvailable());
    enc2.loadModel(temp.resolve("bloom.zip"));
    assertTrue(enc2.isModelAvailable());
    INDArray sent2 = enc2.encode(doc.getSentence(0));
    INDArray word2 = enc2.encode("carolinensis");
    assertFalse(sent2.equals(word2));
    assertEquals(1024, enc2.getEmbeddingVectorSize());
    assertEquals(1024, sent2.length());
    assertEquals(1024, word2.length());
    
    assertEquals(sent1, sent2);
    assertEquals(word1, word2);
    
  }

}
