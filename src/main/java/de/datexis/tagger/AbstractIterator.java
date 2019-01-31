package de.datexis.tagger;

import de.datexis.model.Document;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import de.datexis.model.tag.Tag;
import de.datexis.encoder.EncoderSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author sarnold
 */
public abstract class AbstractIterator implements DataSetIterator {

	protected Logger log = LoggerFactory.getLogger(AbstractIterator.class);

  protected String name;
  protected Collection<Document> documents;
  protected Document currDocument;
  protected List<Document> docsInUse = new ArrayList<>();
  protected EncoderSet encoders;
  protected Class<Tag> tagset;
  
	protected int numExamples;
  protected int totalExamples;
  protected int cursor;
  protected int batchSize;
  protected long inputSize, labelSize;
  protected boolean randomize;
  
  protected long startTime;
  
  /**
   * Create a new Iterator
   * @param name Name of the Dataset
   * @param docs Documents to iterate
   * @param numExamples Number of examples that the iterator should return per epoch
   * @param batchSize Batch size of examples per step
   * @param randomize If True, the document order will be randomized before every epoch
   */
	public AbstractIterator(Collection<Document> docs, String name, int numExamples, int batchSize, boolean randomize) {
    this.name = name;
    this.randomize = randomize;
    this.documents = docs;
    this.totalExamples = (int) docs.stream().mapToInt(d -> d.countSentences()).sum();
    this.numExamples = numExamples < 0 ? this.totalExamples : numExamples;    
    this.batchSize = batchSize;
	}

  @Override
	public abstract void reset();
  
  @Override
	public abstract boolean hasNext();
  
	/**
   * Returns a DataSet with batchSize Documents/Sentences.
   * @return 
   */
  @Override
  public DataSet next() {
    return next(batchSize);
  }
  
	protected void reportProgress() {
    String timeStr = "??";
    try {
      long elapsed = System.currentTimeMillis() - startTime;
      long expected = elapsed * numExamples() / cursor();
      long remaining = expected - elapsed;
      timeStr = String.format("%02d:%02d:%02d",
              TimeUnit.MILLISECONDS.toHours(remaining),
              TimeUnit.MILLISECONDS.toMinutes(remaining) -  
              TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(remaining)),
              TimeUnit.MILLISECONDS.toSeconds(remaining) - 
              TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(remaining)));   
    } catch(Exception e) {
    }
		int progress = (int) ((float) cursor() * 100 / numExamples());
		log.debug("Iterate: returning " + cursor() + "/" + numExamples() + " examples [" + progress + "%, " + timeStr + " remaining]");
	}

  /**
   * Randomizes the order of documents.
   * @param docs
   * @return Iterable with randomized order
   */
  protected List<Document> randomizeDocuments(Collection<Document> docs) {
    log.info("Randomizing documents in " + name + "...");
    List<Document> randomized = new ArrayList<>(docs);
    Collections.shuffle(randomized, new Random(System.nanoTime()));
    return randomized;
  }
  
  public Collection<Document> getDocuments() {
    return documents;
  }
  
  @Override
  public boolean resetSupported() {
    return true;
  }
  
  @Override
  public boolean asyncSupported() {
    return false; // TODO or do we?
  }
  
	@Override
	public int batch() {
		return batchSize;
	}

	public int cursor() {
		return cursor;
	}

  /**
   * Returns the length of the feature vector for one example.
   * @return 
   */
  @Override
  public int inputColumns() {
    // TODO: is this one word or one sentence?
    return (int) inputSize;
  }

  /**
   * Retuns the length of the label vector for one example.
   * @return 
   */
  @Override
  public int totalOutcomes() {
    return (int) labelSize;
  }
  
  public long getInputSize() {
    return inputSize;
  }

  public long getLabelSize() {
    return labelSize;
  }
  
  /**
   * Returns the number of examples this Iterator will return.
   * @return number of documents
   */
  public int numExamples() {
    return numExamples;
  }
  /**
   * Returns the total number of examples in the given Dataset.
   * Note that the number of examples may be set to a different size.
   * @return number of documents
   */
  public int totalExamples() {
    return totalExamples;
  }
  
	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented yet.");
	}
  
  @Override
  public DataSetPreProcessor getPreProcessor() {
    return null;
  }
  
	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented yet.");
	}
  
  public Class getTagset() {
    return tagset;
  };

}
