package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Test;
import static org.junit.Assert.*;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class StructureEncoderTest {
  
  private static final String text = "Perplexity\n" +
    "In information theory, perplexity is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample. Perplexity of a random variable X may be defined as the perplexity of the distribution over its possible values.\n" +
    "In the special case where p models a fair k-sided die (a uniform distribution over k discrete events), k is its perplexity. A random variable with perplexity k has the same uncertainty as a fair k-sided die, and one is said to be \"k-ways perplexed\" about the value of the random variable.\n" +
    "Perplexity is sometimes used as a measure of how hard a prediction problem is.";
  
    private static final String bullets = "Perplexity\n" +
    "- probability to predict a sample\n" +
    "- no more samples. but this may be right.\n" +
    "- last bullet";
  
  public StructureEncoderTest() {
  }
  
  @Test
  public void testTokenEncoding() {
    Document doc = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
    StructureEncoder enc = new StructureEncoder();
    enc.encodeEach(doc, Token.class);
    assertEquals(8, doc.countSentences());
    Sentence s = doc.getSentence(0);
    assertEquals(Nd4j.create(new double[]{1,1,0,1,1,0,0}), s.getToken(0).getVector(StructureEncoder.class)); // Perplexity
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,1,0}), s.getToken(1).getVector(StructureEncoder.class)); // *NL*
    s = doc.getSentence(1);
    assertEquals(Nd4j.create(new double[]{0,1,0,1,0,0,0}), s.getToken(0).getVector(StructureEncoder.class)); // In
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,0,0}), s.getToken(1).getVector(StructureEncoder.class)); // information
    assertEquals(Nd4j.create(new double[]{0,0,0,0,1,0,0}), s.getToken(20).getVector(StructureEncoder.class)); // .
    s = doc.getSentence(2);
    assertEquals(Nd4j.create(new double[]{0,0,0,1,0,0,0}), s.getToken(0).getVector(StructureEncoder.class)); // It
    s = doc.getSentence(4);
    assertEquals(Nd4j.create(new double[]{0,0,0,0,1,0,0}), s.getToken(19).getVector(StructureEncoder.class)); // .
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,1,0}), s.getToken(20).getVector(StructureEncoder.class)); // *NL*
    s = doc.getSentence(7);
    assertEquals(Nd4j.create(new double[]{0,1,0,1,0,0,0}), s.getToken(0).getVector(StructureEncoder.class)); // Perplexity
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,0,0}), s.getToken(13).getVector(StructureEncoder.class)); // is
    assertEquals(Nd4j.create(new double[]{0,0,0,0,1,1,1}), s.getToken(14).getVector(StructureEncoder.class)); // .
  }
  
  @Test
  public void testSentenceEncoding() {
    Document doc = DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
    StructureEncoder enc = new StructureEncoder();
    enc.encodeEach(doc, Sentence.class);
    assertEquals(8, doc.countSentences());
    assertEquals(Nd4j.create(new double[]{1,1,0,0,0,1,0}), doc.getSentence(0).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,0,0,0,0,0}), doc.getSentence(1).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,0,0}), doc.getSentence(2).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,0,0}), doc.getSentence(3).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,1,0}), doc.getSentence(4).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,0,0,0,0,0}), doc.getSentence(5).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,1,0}), doc.getSentence(6).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,0,0,0,1,1}), doc.getSentence(7).getVector(StructureEncoder.class));
  }
  
  @Test
  public void testListEncoding() {
    Document doc = DocumentFactory.fromText(bullets, DocumentFactory.Newlines.KEEP);
    StructureEncoder enc = new StructureEncoder();
    enc.encodeEach(doc, Sentence.class);
    assertEquals(5, doc.countSentences());
    assertEquals(Nd4j.create(new double[]{1,1,0,0,0,1,0}), doc.getSentence(0).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,1,0,0,1,0}), doc.getSentence(1).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,1,0,0,0,0}), doc.getSentence(2).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,0,0,0,0,1,0}), doc.getSentence(3).getVector(StructureEncoder.class));
    assertEquals(Nd4j.create(new double[]{0,1,1,0,0,1,1}), doc.getSentence(4).getVector(StructureEncoder.class));
  }

}
