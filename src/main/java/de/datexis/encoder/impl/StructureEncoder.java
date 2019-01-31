package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.slf4j.LoggerFactory;

/**
 * Encodes structural features, such as BOD (Begin of Document), BOS (Begin of Sentence), NL (Newline) etc.
 * @author sarnold
 */
public class StructureEncoder extends StaticEncoder {
  
  public StructureEncoder() {
    super("STR");
    log = LoggerFactory.getLogger(StructureEncoder.class);
  }
  
  public StructureEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(StructureEncoder.class);
  }
  
  @Override
  public String getName() {
    return "Structure Encoder";
  }

  @Override
  @JsonIgnore
  public long getEmbeddingVectorSize() {
    return encode("Test").length();
  }
  
  public void setVectorSize(int size) {
    if(size != getEmbeddingVectorSize()) {
      throw new IllegalArgumentException("Vector size of saved Encoder (" + getEmbeddingVectorSize() + ") differs from implementation (" + size + ")");
    }
  }

  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    INDArray encoding = Nd4j.zeros(input.size(), getEmbeddingVectorSize(), maxTimeSteps);
    Document example;

    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      
      example = input.get(batchIndex);
      if(timeStepClass.equals(Token.class)) {
        List<INDArray> vecs = encodeTokens(example);
        for(int t = 0; t < example.countTokens() && t < maxTimeSteps; t++) {
          encoding.getRow(batchIndex).getColumn(t).assign(vecs.get(t));
        }
      } else if(timeStepClass.equals(Sentence.class)) {
        List<INDArray> vecs = encodeSentences(example);
        for(int t = 0; t < example.countSentences() && t < maxTimeSteps; t++) {
          encoding.getRow(batchIndex).getColumn(t).assign(vecs.get(t));
        }
      } else throw new IllegalArgumentException("Cannot encode class " + timeStepClass.toString() + " from Document");
      
    }
    return encoding;
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }
  
  @Override
  public INDArray encode(String token) {
    return createVector(false, false, false, false, false, false, false);
  }
  
  protected INDArray createVector(boolean beginDoc, boolean beginParagraph, boolean beginSent, boolean endSent, boolean endParagraph, boolean endDoc, boolean isList) {
    return Nd4j.create(new double[] {
      beginDoc ? 1.0 : 0.0,       // begin of document
      beginParagraph ? 1.0 : 0.0, // begin of paragraph
      isList ? 1.0 : 0.0,         // sentence is part of list
      beginSent ? 1.0 : 0.0,      // begin of sentence
      endSent ? 1.0 : 0.0,        // end of sentence
      endParagraph ? 1.0 : 0.0,   // end of paragraph
      endDoc ? 1.0 : 0.0          // end of document
    });
  }
  
  @Override
  public void encodeEach(Document d, Class<? extends Span> timeStepClass) {
    int i = 0;
    if(timeStepClass == Token.class) {
      List<INDArray> vecs = encodeTokens(d);
      for(Token t : d.getTokens()) t.putVector(StructureEncoder.class, vecs.get(i++));
    } else if(timeStepClass == Sentence.class) {
      List<INDArray> vecs = encodeSentences(d);
      for(Sentence s : d.getSentences()) s.putVector(StructureEncoder.class, vecs.get(i++));
    } else throw new IllegalArgumentException("Cannot encode class " + timeStepClass.toString() + " from Document");
  }

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    throw new IllegalArgumentException("StructureEncoder is only implemented to encode over Documents.");
  }
  
  private List<INDArray> encodeTokens(Document d) {
    List<INDArray> result = new ArrayList<>(d.countTokens());
    boolean beginDoc = true, lastWasNL = true, isNL, isNextNL, beginSent, endSent, endDoc, isList;
    Iterator<Sentence> sentences = d.getSentences().iterator();
    while(sentences.hasNext()) {
      Sentence s = sentences.next();
      endDoc = !sentences.hasNext();
      beginSent = true;
      Iterator<Token> tokens = s.getTokens().iterator();
      int i =0;
      while(tokens.hasNext()) {
        Token t = tokens.next();
        Token n = tokens.hasNext() ? s.getToken(i + 1) : null;
        endSent = n == null;
        isList = beginSent && t.getText().equals("-");
        isNL = t.getText().equals("*NL*") || t.getText().equals("\n");
        isNextNL = n != null && (n.getText().equals("*NL*") || n.getText().equals("\n"));
        result.add(createVector(beginDoc && beginSent, lastWasNL && beginSent, beginSent, (endSent && !isNL) || isNextNL, isNL || (endDoc && endSent), endDoc && endSent, isList));
        beginSent = false;
        lastWasNL = isNL;
        i++;
      }
      beginDoc = false;
    }
    return result;
  }
  
  private List<INDArray> encodeSentences(Document d) {
    List<INDArray> result = new ArrayList<>(d.countSentences());
    boolean beginDoc = true, beginPar = true, endPar, endDoc, isList;
    Iterator<Sentence> sentences = d.getSentences().iterator();
    while(sentences.hasNext()) {
      Sentence s = sentences.next();
      endDoc = !sentences.hasNext();
      endPar = s.streamTokens().anyMatch(t -> t.getText().equals("*NL*") || t.getText().equals("\n"));
      isList = s.getText().startsWith("- ");
      result.add(createVector(beginDoc, beginPar || beginDoc, false, false, endPar || endDoc, endDoc, isList));
      beginDoc = false;
      beginPar = endPar;
    }
    return result;
  }
  
}
