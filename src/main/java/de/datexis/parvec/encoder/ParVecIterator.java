package de.datexis.parvec.encoder;

import de.datexis.model.*;
import de.datexis.sector.model.SectionAnnotation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An iterator that iterates over Sections (not Sentences).
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ParVecIterator implements LabelAwareSentenceIterator {

  protected final static Logger log = LoggerFactory.getLogger(ParVecIterator.class);

  protected List<Document> documents;
  protected Iterator<Document> docIt;
  protected Document currentDoc;
  protected List<SectionAnnotation> sections;
  protected Iterator<SectionAnnotation> annIt;
  protected SectionAnnotation currentSection;
  
  protected int batchSize;
  protected boolean randomize;
  
  public ParVecIterator(Dataset data, boolean randomize) {
    this(data.getDocuments(), randomize);
  }

  public ParVecIterator(Collection<Document> docs, boolean randomize) {
    this.documents = new ArrayList<>(docs);
    this.randomize = randomize;
  }
  
  @Override
  public void reset() {
    currentSection = null;
    annIt = null;
    sections = new ArrayList<>(64);
    currentDoc = null;
    if(randomize) Collections.shuffle(documents, new Random(System.nanoTime()));
    docIt = documents.iterator();
  }
  
  @Override
  public boolean hasNext() {
    if(hasNextSection()) {
      return true;
    } else if(hasNextDocument()) {
      nextDocument();
      return hasNext();
    } else {
      return false;
    }
  }
  
  protected boolean hasNextSection() {
    return annIt != null && annIt.hasNext();
  }
  
  protected boolean hasNextDocument() {
    return docIt != null && docIt.hasNext();
  }
  
  public void nextDocument() {
    currentDoc = docIt.next();
    sections = currentDoc.streamAnnotations(Annotation.Source.GOLD, SectionAnnotation.class).collect(Collectors.toList());
    //log.trace("returning doc {} with {} sections.", currentDoc.getId(), sections.size());
    annIt = sections.iterator();
  }
  
  @Override
  public String nextSentence() {
    currentSection = annIt.next();
    String text = currentDoc
        .streamTokensInRange(currentSection.getBegin(), currentSection.getEnd(), true)
        .map(t -> t
            .getText()
            .trim()
            .replaceAll("\n", "*NL*")
            .replaceAll("\t", "*t*"))
        .collect(Collectors.joining(" "));
    return text;
  }

  @Override
  public String currentLabel() {
    return currentSection.getSectionLabel();
  }

  @Override
  public List<String> currentLabels() {
    return Arrays.asList(currentSection.getSectionLabel());
  }

  @Override
  public void finish() {
    currentSection = null;
    annIt = null;
    sections.clear();
    currentDoc = null;
    docIt = null;
    documents.clear();
  }

  @Override
  public SentencePreProcessor getPreProcessor() {
    return null;
  }

  @Override
  public void setPreProcessor(SentencePreProcessor preProcessor) {
  }
  
}
