package de.datexis.encoder;


import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.model.Document;
import de.datexis.preprocess.DocumentFactory;
import java.io.IOException;
import java.util.Arrays;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class BagOfWordsEncoderTest {
  
  final private String text = "Can a movie actually convince you to support torture? Can a movie really persuade you that \"fracking\" -- a process used to drill for "
          + "natural gas -- is a danger to the environment? Can a movie truly cause you to view certain minority groups in a negative light? Some scoff at the notion "
          + "that movies do anything more than entertain. They are wrong. Sure, it's unlikely that one movie alone will change your views on issues of magnitude. But "
          + "a movie (or TV show) can begin your \"education\" or \"miseducation\" on a topic. And for those already agreeing with the film's thesis, it can further "
          + "entrench your views. Anyone who doubts the potential influence that movies can have on public opinion need to look no further than two films that are "
          + "causing an uproar even before they have opened nationwide. They present hot button issues that manage to fire up people from the left and right. The "
          + "first, \"Zero Dark Thirty,\" is about the pursuit and killing of Osama bin Laden, which features scenes of torture. The second, \"Promised Land,\" stars "
          + "Matt Damon and explores how the use of fracking to drill for natural gas can pose health and environmental dangers. Critics of \"Zero Dark Thirty\" fear "
          + "that audiences will accept as true the film's story line that torture was effective in eliciting information to locate bin Laden. They are rightfully "
          + "concerned that the film will sway some to become more receptive or even supportive of the idea of torturing prisoners. Peter Bergen: Did torture really "
          + "net bin Laden Opposition to the film escalated last week as three senior U.S. senators -- John McCain, Carl Levin and Dianne Feinstein -- sent a letter "
          + "to the film's distributor, Sony Pictures, characterizing the film's use of torture as \"grossly inaccurate and misleading.\" The senators bluntly informed "
          + "Sony Pictures that it has \"an obligation to state that the role of torture in the hunt for Osama bin Laden is not based on the facts, but rather part of "
          + "the film's fictional narrative.\"";
  
  public BagOfWordsEncoderTest() {
  }
  
  @Test
  public void testNHotEncoder() {
    Document doc = DocumentFactory.fromText(text);
    BagOfWordsEncoder enc = new BagOfWordsEncoder();
    assertFalse(enc.isModelAvailable());
    enc.trainModel(Arrays.asList(doc), 3, WordHelpers.Language.EN);
    assertTrue(enc.isModelAvailable());
    assertTrue(enc.getEmbeddingVectorSize() >= 8);
    assertEquals(6, enc.getFrequency("torture"));
    assertFalse(enc.isUnknown("torture"));
    assertEquals(0, enc.getFrequency("concerned")); // 1 < minFrequency
    assertTrue(enc.isUnknown("concerned"));
    assertEquals(4, enc.getFrequency("laden"));
    assertFalse(enc.isUnknown("laden"));
    assertEquals(4, enc.getFrequency("Laden"));
    assertFalse(enc.isUnknown("Laden"));
    assertEquals(0, enc.getFrequency("the")); // stopword
    assertTrue(enc.isUnknown("the"));
  }
  
  @Test
  public void testEncodings() {
    Document doc = DocumentFactory.fromText(text);
    BagOfWordsEncoder enc = new BagOfWordsEncoder();
    enc.trainModel(Arrays.asList(doc), 3, WordHelpers.Language.EN);
    INDArray a = enc.encode("laden");
    // this has to pass for all Encoders. Don't change!
    long size = enc.getEmbeddingVectorSize();
    assertEquals(size, a.length());
    assertEquals(size, a.size(0));
    assertEquals(size, a.rows());
    assertEquals(1, a.size(1));
    assertEquals(1, a.columns());
    assertEquals(2, a.rank());
  }
  
  @Test
  public void saveLoadEncoderTest() throws IOException {
    Document doc = DocumentFactory.fromText(text);
    BagOfWordsEncoder enc = new BagOfWordsEncoder();
    enc.trainModel(Arrays.asList(doc), 3, WordHelpers.Language.EN);
    int idx = enc.getIndex("blubb");
    assertEquals(-1, idx);
    idx = enc.getIndex("laden");
    assertTrue(idx >= 0);
    String word = enc.getWord(idx);
    assertEquals("laden", word);
    
    Resource temp = Resource.createTempDirectory();
    enc.saveModel(temp, "bow");
    BagOfWordsEncoder enc2 = new BagOfWordsEncoder();
    enc2.loadModel(temp.resolve("bow.tsv.gz"));
    assertEquals(enc.getEmbeddingVectorSize(), enc2.getEmbeddingVectorSize());
    idx = enc2.getIndex("blubb");
    assertEquals(-1, idx);
    idx = enc2.getIndex("laden");
    assertTrue(idx >= 0);
    word = enc2.getWord(idx);
    assertEquals("laden", word);
  }
  
  @Test
  public void testNearestNeighbours() {
    Document doc = DocumentFactory.fromText(text);
    BagOfWordsEncoder enc = new BagOfWordsEncoder();
    enc.trainModel(Arrays.asList(doc), 3, WordHelpers.Language.EN);
    INDArray a = enc.encode("laden");
    INDArray b = enc.encode("torture");
    assertEquals("laden", enc.getNearestNeighbour(a));
    assertEquals("[laden]", enc.getNearestNeighbours(a, 1).toString());
    assertEquals("torture", enc.getNearestNeighbour(b));
    INDArray c = a.mul(0.6).add(b.mul(0.2));
    assertEquals("laden", enc.getNearestNeighbour(c));
    assertEquals("[laden]", enc.getNearestNeighbours(c, 1).toString());
    assertArrayEquals(new String[] { "laden","torture" }, enc.getNearestNeighbours(c, 2).toArray());
  }
  
}
