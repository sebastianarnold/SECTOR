package de.datexis.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import java.util.ArrayList;
import java.util.List;
import de.datexis.common.WordHelpers;
import static de.datexis.model.Dataset.random;
import de.datexis.model.tag.Tag;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.preprocess.DocumentFactory.Newlines;
import java.util.Collection;
import java.util.EnumMap;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A Document is a piece of text that mayu contain Sentences, Tokens and Annotations.
 * @author sarnold, fgrimme
 */
@JsonPropertyOrder({ "class", "id", "uid", "refUid", "title", "language", "type", "begin", "length", "text", "annotations" })
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "class", defaultImpl=Document.class)
@JsonIgnoreProperties(ignoreUnknown = true)
public class Document extends Span {

	/**
	 * All Sentences the Document is referencing
	 */
	protected List<Sentence> sentences;
  
  /**
   * List of Annotations that were assigned to this Span from Gold, Prediction or User sources.
   * Only initialized when used.
   */
  protected List<Annotation> annotations;
  
  /**
   * The ID of this document (e.g. URL)
   */
  private String id = null;
  
  /**
   * The language of this document
   */
  private String language = null;
  
  /**
   * The type of this document
   */
  private String type = null;
  
  /**
   * The title of this document
   */
  private String title = null;
  
  /**
   * An arbitrary document source that can be used for provenance.
   */
  private Object source = null;
  
  /**
   * List of Tags that were assigned to this Dataset Gold, Prediction or User sources.
   * Only initialized when used.
   */
  private EnumMap<Annotation.Source, TreeSet<String>> assignedTags = null;
  
	/**
	 * Create an empty Document
	 */
	public Document() {
		sentences = new ArrayList<>();
    annotations = new ArrayList<>();
	}
  
	/**
	 * Set sentences
	 * @param s new value for sentences of this Document instance
	 */
  @Deprecated
	public void setSentences(List<Sentence> s) {
    if(!s.isEmpty()) {
      sentences = s;
      begin = sentences.get(0).getBegin();
      end = sentences.get(sentences.size() - 1).getEnd();
    } else {
      sentences = new ArrayList<>();
      begin = 0;
      end = 0;
    }
	}
  
  /**
	 * @return all Sentences in this Document
	 */
  @JsonIgnore
	public List<Sentence> getSentences() {
		return sentences;
	}
  
  /**
	 * @return Stream all Sentences in this Document
	 */
  @JsonIgnore
	public Stream<Sentence> streamSentences() {
		return sentences.stream();
	}
  
  /**
   * Returns a single Sentence from this Document
   * @param offset position (number of sentences) starting at 0
   * @return 
   */
  public Sentence getSentence(int offset) {
    return sentences.get(offset);
  }
  
  /**
   * Returns the complete Sentence at a given Span position.
   * @return Sentence or NULL if no sentence was found at that position
   */
  public Optional<Sentence> getSentenceAtPosition(int begin) {
    return getSentences().stream()
            .filter(s -> (s.getBegin() <= begin && s.getEnd() > begin))
            .findFirst();
  }
  
  /**
   * Returns the Sentence index at a given Span position.
   * @return index or -1 if no sentence was found at that position
   */
  public int getSentenceIndexAtPosition(int begin) {
    AtomicInteger index = new AtomicInteger(-1);
    Optional<Sentence> sentence = getSentences().stream()
            .peek(s -> index.incrementAndGet())  // increment every element encounter
            .filter(s -> (s.getBegin() > begin)) // find first sentence that starts later
            .findFirst();
    if(sentence.isPresent()) return index.get() - 1;
    else return index.get();
  }
  
  /**
   * @return all Sentences that are in a given range.
   * @param begin
   * @param end
   * @param enclosed - TRUE to return only completely enclosed sentences, FALSE to expand sentences at the boundaries
   */
  public Stream<Sentence> streamSentencesInRange(int begin, int end, boolean enclosed) {
    if(enclosed) return getSentences().stream()
            .filter(t -> t.getBegin() >= begin && t.getEnd() <= end);
    else return getSentences().stream()
            .filter(t -> (t.getBegin() <= begin && t.getEnd() > begin) ||
                         (t.getBegin() >= begin && t.getEnd() <= end && begin != end) || 
                         (t.getBegin() < end && t.getEnd() >= end));
  }
  
  /**
   * @return all Tokens in a given range.
   * @param enclosed - TRUE to return only completely enclosed tokens, FALSE to expand sentences at the boundaries
   */
  public Stream<Token> streamTokensInRange(int begin, int end, boolean enclosed) {
    if(enclosed) return streamTokens().filter(t -> t.getBegin() >= begin && t.getEnd() <= end);
    else return streamTokens().filter(t -> (t.getBegin() <= begin && t.getEnd() > begin) || 
                                           (t.getBegin() >= begin && t.getEnd() <= end && begin != end) || 
                                           (t.getBegin() < end && t.getEnd() >= end));
  }

  public Optional<Token> getToken(int pos) {
    return streamTokens().skip(pos).limit(1).findFirst();
  }
  
  /**
   * Returns a random Sentence from this Document
   * @return 
   */
  @JsonIgnore
  Sentence getRandomSentence() {
    int index = random.nextInt(sentences.size());
    return getSentence(index);
  }
  
  /**
   * Appends a Sentence to the end of the document. Span offsets are adjusted accordingly.
   * @param s The Sentence to add.
   */
	public void addSentence(Sentence s) {
    addSentence(s, true);
	}
  
  public void addSentence(Sentence s, boolean adjustOffsets) {
    if(adjustOffsets) {
      if(sentences.isEmpty()) begin = 0;
      int cursor = getEnd();
      if (!sentences.isEmpty()) cursor ++;
      int length = s.getLength();
      // FIXME: setBegin should adjust Token's positions. Or use relative positions thoughout.
      s.setBegin(cursor);
      s.setLength(length);
      end = s.getEnd();
    } else {
      if(sentences.isEmpty()) begin = s.getBegin();
      end = s.getEnd();
    }
    s.setDocumentRef(this);
		sentences.add(s);
  }
  
  /*
  we simply use .endswith, hope that is not too slow
  public void append(Document doc) {
    if(isEmpty()) append(doc, false);
    else if(streamTokens() // check if last token is newline
            .reduce((a, b) -> b) 
            .filter(t -> t.getText().equals("\n"))
            .isPresent()
            ) append(doc, false); 
    else append(doc, true);
  }*/
  
  public void append(Document doc) {
    int offset;
    if(isEmpty() || getText().endsWith("\n") || getText().endsWith(" ")) offset = getEnd();
    else offset = getEnd() + 1;
    int length = doc.getLength();
    doc.setBegin(doc.getBegin() + offset);
    doc.setLength(length);
    for(Sentence s : doc.getSentences()) {
      s.addOffset(offset);
      s.setDocumentRef(this);
      sentences.add(s);
      setEnd(s.getEnd());
      doc.setEnd(s.getEnd());
    }
  }
  
  public void setId(String id) {
    this.id = id;
  }
  
  public String getId() {
    return this.id;
  }
  
  public <T extends Tag> void setTagAvailable(Annotation.Source source, Class<T> tag, boolean exists) {
    if(exists) { // set
      if(assignedTags == null) assignedTags = new EnumMap<>(Annotation.Source.class);
      if(!assignedTags.containsKey(source)) assignedTags.put(source, new TreeSet<>());
      assignedTags.get(source).add(tag.getCanonicalName());
    } else { //unset
      if(assignedTags != null && assignedTags.containsKey(source)) {
        assignedTags.get(source).remove(tag.getCanonicalName());
      }
    }
  }
  
  public <T extends Tag> boolean isTagAvaliable(Annotation.Source source, Class<T> tag) {
    if(assignedTags != null && assignedTags.containsKey(source)) {
      return assignedTags.get(source).contains(tag.getCanonicalName());
    } else {
      return false;
    }
  }
  
  /**
   * Adds a single Annotation to this document.
   * @param <A> Type of the Annotation
   */
  public <A extends Annotation> void addAnnotation(A ann) {
    if(annotations == null) annotations = new ArrayList<>(countSentences() * 4);
    ann.setDocumentRef(this);
    annotations.add(ann);
  }
 
  /**
   * Adds a List of Annotations to this document.
   * @param ann The Annotations to add. Duplicates will not be replaced.
   */
  public void addAnnotations(List<? extends Annotation> anns) {
    if(annotations == null) annotations = new ArrayList<>(Math.max(countSentences() * 4, anns.size() * 2));
    anns.stream().forEach(ann -> ann.setDocumentRef(this));
    annotations.addAll(anns);
  }
  
  /**
   * @return ALL Annotations attached to this Document.
   */
  @JsonIgnore
  protected Stream<? extends Annotation> streamAnnotations() {
    if(annotations == null) return Stream.empty();
    else return annotations.stream().map(ann -> ann.getClass().cast(ann));
  }
  
  public <A extends Annotation> Stream<A> streamAnnotations(Annotation.Source source, Class<A> type) {
    return streamAnnotations()
            .filter(ann -> ann.getClass().equals(type) && ann.source.equals(source))
            .map(ann -> (A) ann);
  }
  
  public <A extends Annotation> Stream<A> streamAnnotations(Class<A> type) {
    return streamAnnotations()
            .filter(ann -> ann.getClass().equals(type))
            .map(ann -> (A) ann);
  }
  
  public Stream<? extends Annotation>  streamAnnotations(Annotation.Source source) {
    return streamAnnotations()
            .filter(ann -> ann.source.equals(source));
  }

  public <A extends Annotation> Stream<A> streamAnnotationsIncludingSubtypes(Class<A> type) {
    return streamAnnotations()
      .filter(ann -> type.isAssignableFrom(ann.getClass()))
      .map(ann -> (A) ann);
  }

  /**
   * @return all Annotations for JSON deserialization
   */
  public Collection<? extends Annotation> getAnnotations() {
    return streamAnnotations().collect(Collectors.toList());
  }
  
  /**
   * Returns all matching Annotations in a given range
   * @param source Origin of the Annotation
   * @param type return only Annotations of the requested Class
   * @param begin return only Annotations in the given range
   * @param end return only Annotations in the given range
   * @param enclosed TRUE to return only completely enclosed Annotations, FALSE to return all Annotations that intersect
   * @return
   */
  public <A extends Annotation> Stream<A> streamAnnotationsInRange(Annotation.Source source, Class<A> type, int begin, int end, boolean enclosed) {
    if(enclosed) return streamAnnotations(source, type)
            .filter(a -> a.getBegin() >= begin && a.getEnd() <= end);
    else return streamAnnotations(source, type)
            .filter(a -> (begin <= a.getBegin() && end > a.getBegin()) ||
                         (begin >= a.getBegin() && end <= a.getEnd() && begin != end) || 
                         (begin < a.getEnd() && end >= a.getEnd()));
  };
  
  public <A extends Annotation> Stream<A> streamAnnotationsForSpan(Annotation.Source source, Class<A> type, Span s) {
    return streamAnnotationsInRange(source, type, s.getBegin(), s.getEnd(), false);
  }
  
  public <A extends Annotation> Collection<A> getAnnotationsForSpan(Annotation.Source source, Class<A> type, Span s) {
    return streamAnnotationsForSpan(source, type, s).collect(Collectors.toList());
  }
  
  /**
   * Returns the annotation that has the largest overlapping range
   */
  public <A extends Annotation> Optional<A> getAnnotationMaxOverlap(Annotation.Source source, Class<A> type, Span s) {
    Stream<A> anns = streamAnnotationsInRange(source, type, s.getBegin(), s.getEnd(), false); // all intersecting annotations
    return anns.reduce((first,second) -> // find maximum overlapping range
      WordHelpers.getSpanOverlapLength(s,second) > WordHelpers.getSpanOverlapLength(s,first) ? second : first
    );
  }
  
  public <A extends Annotation> Collection<A> getAnnotations(Class<A> type) {
    return streamAnnotations(type).collect(Collectors.toList());
  };
  
  public Collection<? extends Annotation> getAnnotations(Annotation.Source source) {
    return streamAnnotations(source).collect(Collectors.toList());
  };
  
  public <A extends Annotation> Collection<A> getAnnotations(Annotation.Source source, Class<A> type) {
    return streamAnnotations(source, type).collect(Collectors.toList());
  };
  
  public long countAnnotations(Annotation.Source source) {
    return streamAnnotations(source).count();
  }
   
  public <A extends Annotation> long countAnnotations(Class<A> type) {
    return streamAnnotations(type).count();
  }
  
  public <A extends Annotation> long countAnnotations(Annotation.Source source, Class<A> type) {
    return streamAnnotations(source, type).count();
  }
  
  public <A extends Annotation> void clearAnnotations(Annotation.Source source, Class<A> type) {
    if(annotations != null) annotations.removeIf(ann -> ann.getClass().equals(type) && ann.source.equals(source));
  }

  public boolean removeAnnotation(Annotation annotationToRemove) {
    return annotations.remove(annotationToRemove);
  }
  
	/**
	 * @return stream over all Tokens in this Document
	 */
  @JsonIgnore
	public Stream<Token> streamTokens() {
    return streamSentences().flatMap(s -> s.streamTokens());
	}
  
  /**
	 * @return collection of all Tokens in this Document
	 */
  @JsonIgnore
	public List<Token> getTokens() {
    return streamTokens().collect(Collectors.toList());
	}
  
  @JsonIgnore
  public <S extends Span> Stream<S> getStream(Class<S> spanClass) {
    if(spanClass == Sentence.class) return (Stream<S>) streamSentences();
    else if(spanClass == Token.class) return (Stream<S>) streamTokens();
    else return (Stream<S>) streamTokens();
  }
  
	/**
	 * @return the whole Document text as String
	 */
  @Override
	public String getText() {
    if(countTokens() > 0) {
      return WordHelpers.tokensToText(getTokens(), getBegin());
    } else if(countSentences() > 0) {
      StringBuilder res = new StringBuilder();
      for(Sentence s : getSentences()) {
        res.append(s.getText());
        res.append(" ");
      }
      return res.toString();
    } else {
      return "";
    }
	}
  
  /**
   * @return the number of sentences in the Document
   */
  public int countSentences() {
    return sentences.size();
  }
  
  /**
   * @return the number of tokens in the Document
   */
  public int countTokens() {
    return (int) streamTokens().count();
  }
  
  public long countAnnotations() {
    if(annotations == null) return 0;
    else return (int) annotations.size();
  }
  
  /**
   * @return TRUE if the document is empty
   */
  @JsonIgnore
  public boolean isEmpty() {
    return !streamTokens().findFirst().isPresent();
  }

  @Override
	public String toString() {
		return "Document [sentences=" + sentences + "]";
	}

  /**
   * Sets an abitrary Object as source. Can be used to store the original 
   * data structure for provenance information.
   * @param src 
   */
  public void setSource(Object src) {
    source = src;
  }

  /**
   * @return the attached original source Object
   */
  @JsonIgnore
  public Object getSource() {
    return source;
  }

  public String getLanguage() {
    return language;
  }

  public void setLanguage(String language) {
    this.language = language;
  }
  
  public String getType() {
    return type;
  }

  public void setType(String type) {
    this.type = type;
  }

  public String getTitle() {
    return title;
  }

  public void setTitle(String title) {
    this.title = title;
  }
  
  /**
   * Used for JSON Deserialization
   * @param text
   */
  public void setText(String text) {
    sentences.clear();
    setBegin(0);
    setEnd(0);
    DocumentFactory.getInstance().addToDocumentFromText(text, this, Newlines.KEEP);
  }

  /**
   * @return a deep copy of this Document
   */
  public Document clone() {
    return clone(Function.identity());
  }

  public Document clone(Function<String,String> transform) {
    Document result = new Document();
    for(Sentence s : getSentences()) {
      result.addSentence(s.clone(transform), false);
    }
    result.setId(getId());
    result.setBegin(getBegin());
    result.setEnd(getEnd());
    result.setSource(getSource());
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if(this == o) {
      return true;
    }
    if(!(o instanceof Document)) {
      return false;
    }
    Document document = (Document) o;
    return super.equals(document) &&
           Objects.equals(getLanguage(), document.getLanguage()) &&
           Objects.equals(getType(), document.getType()) &&
           Objects.equals(getSentences(), document.getSentences()) &&
           Objects.equals(getAnnotations(), document.getAnnotations()) &&
           Objects.equals(assignedTags, document.assignedTags);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), getSentences(), getAnnotations(),
                        getLanguage(), getType(), assignedTags);
  }
}
