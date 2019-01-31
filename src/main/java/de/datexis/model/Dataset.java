package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Holds a collection of Documents in memory(!)
 * @author sarnold
 */
public class Dataset {
  
  private static final Logger log = LoggerFactory.getLogger(Dataset.class);
  
  private String name;
  private String language = null;
  
  /**
   * The unique ID of this span (e.g. database primary key)
   */
  protected Long uid = null;
  
  private List<Document> documents;
  public static Random random = new Random();
  
  public Dataset(String name) {
    documents = new ArrayList<>();
    this.name = name;
  }
  
  public Dataset(String name, List<Document> docs) {
    this.documents = docs;
    this.name = name;
  }
  
  /**
   * Default constructor.
   */
  public Dataset() {
    this("");
  }
  
  public String getName() {
    return name;
  }
  
  public void setName(String name) {
    this.name = name;
  }

  public String getLanguage() {
    return language;
  }
  
  public void setUid(Long uid) {
    this.uid = uid;
  }
  
  @JsonInclude(JsonInclude.Include.NON_NULL)
  public Long getUid() {
    return this.uid;
  }

  public void setLanguage(String language) {
    this.language = language;
  }
  
  /**
   * Create a Dataset that references to a split of documents. Caution: this is not a deep copy.
   * @param offset
   * @param count
   * @return 
   */
  public Dataset getSplit(int offset, int count) {
    if(offset < 0) offset = countDocuments() + offset;
    if(count < 0) count = countDocuments() + count;
    try {
      List<Document> docs = documents.subList(offset, offset + count);
      return new Dataset(getName(), docs);
    } catch(IndexOutOfBoundsException ex) {
      log.warn("Document index out of bounds, returning whole dataset " + ex);
      return new Dataset(getName(), documents);
    }
  }
  
  /**
   * Iterate over all Documents in this Dataset
   * @return 
   */
  public List<Document> getDocuments() {
    return documents;
  }
  
  public Stream<Document> streamDocuments() {
    return documents.stream();
  }
  
  /**
   * Iterate over a subset of Documents in this Dataset
   * @param offset
   * @param count
   * @return 
   */
  public List<Document> getDocuments(int offset, int count) {
    return streamDocuments(offset, count).collect(Collectors.toList());
  }
  
  public Stream<Document> streamDocuments(int offset, int count) {
    return streamDocuments().skip(offset).limit(count);
  }
  
  public Optional<Document> getDocument(int offset) {
     return streamDocuments().skip(offset).findFirst();
  }
  
  /**
   * Return a random Document of this Dataset
   * @return 
   */
  @JsonIgnore
  public Optional<Document> getRandomDocument() {
    int index = random.nextInt(documents.size());
    return getDocument(index);
  }

  public void randomizeDocuments() {
    Collections.shuffle(documents);
  }
  
  public void randomizeDocuments(long seed) {
    Collections.shuffle(documents, new Random(seed));
  }
  
  /**
   * @return stream over all Sentences in the Dataset. Caution: Boundaries are still given on Document level.
   */
  @JsonIgnore
  public Stream<Sentence> streamSentences() {
		return streamDocuments().flatMap(s -> s.streamSentences());
  }
  
  /**
   * @return stream over all Tokens in the Dataset. Caution: Boundaries are still given on Document level.
   */
  @JsonIgnore
  public Stream<Token> streamTokens() {
		return streamDocuments().flatMap(s -> s.streamTokens());
  }
  
  @JsonIgnore
  public <S extends Span> Stream<S> getStream(Class<S> spanClass) {
    if(spanClass == Sentence.class) return (Stream<S>) streamSentences();
    else if(spanClass == Token.class) return (Stream<S>) streamTokens();
    else return (Stream<S>) streamTokens();
  }
  
  /**
   * Add a document to the end of this Dataset
   * @param doc 
   */
  public void addDocument(Document doc) {
    if(language == null) setLanguage(doc.getLanguage());
    documents.add(doc);
  }
  
  public void addDocumentFront(Document d) {
    documents.add(0, d);
  }
  
  /**
   * Returns the number of Documents in this Dataset
   * @return 
   */
  public int countDocuments() {
    return documents.size();
  }
  
  /**
   * @return the number of Sentences in all Documents in this Dataset
   */
  public long countSentences() {
    return streamDocuments().mapToLong(d -> d.countSentences()).sum();
  }
  
  /**
   * @return the number of Tokens in all Documents in this Dataset
   */
  public long countTokens() {
    return streamDocuments().mapToLong(d -> d.countTokens()).sum();
  }
  
  /**
   * @return the number of Annotations in all Documents in this Dataset
   */
  public long countAnnotations() {
    return streamDocuments().mapToLong(d -> d.countAnnotations()).sum();
  }
  
  public long countAnnotations(Annotation.Source source) {
    return streamDocuments().mapToLong(d -> d.countAnnotations(source)).sum();
  }
  
  @JsonIgnore
  public Sentence getRandomSentence() {
    int index = random.nextInt(documents.size());
    return documents.get(index).getRandomSentence();
  }
  
  /**
   * @return a deep copy of this Dataset
   */
  public Dataset clone() {
    ArrayList<Document> docs = new ArrayList<>(countDocuments());
    for(Document doc : getDocuments()) {
      docs.add(doc.clone());
    }
    return new Dataset(getName(), docs);
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Dataset)) {
      return false;
    }
    
    Dataset dataset = (Dataset) o;
    return Objects.equals(getName(), dataset.getName()) &&
           Objects.equals(getLanguage(), dataset.getLanguage()) &&
           Objects.equals(getDocuments(), dataset.getDocuments());
  }

  @Override
  public int hashCode() {
    return Objects.hash(getName(), getLanguage(), getDocuments());
  }
}