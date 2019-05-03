package de.datexis.reader;

import de.datexis.common.Resource;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads all files from a directory into a Dataset.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class RawTextDatasetReader implements DatasetReader {

  protected final static Logger log = LoggerFactory.getLogger(RawTextDatasetReader.class);
  
  protected boolean randomizeDocuments = false;
  protected boolean useFirstSentenceAsTitle = false;
  protected boolean isTokenized = false;
  protected long limit = -1;
  
  /**
   * Use a copy of every first sentence as Document title.
   */
  public RawTextDatasetReader withRandomizedDocuments(boolean randomize) {
    this.randomizeDocuments = randomize;
    return this;
  }
  
  /**
   * Use a copy of every first sentence as Document title.
   */
  public RawTextDatasetReader withFirstSentenceAsTitle(boolean useFirstSentence) {
    this.useFirstSentenceAsTitle = useFirstSentence;
    return this;
  }
  
  /**
   * Stop after reading a given number of documents.
   */
  public RawTextDatasetReader withLimitNumberOfDocuments(long limit) {
    this.limit = limit;
    return this;
  }
  
  /**
   * Set to TRUE if the input files are already tokenized and space-separated.
   */
  public RawTextDatasetReader withTokenizedInput(boolean isTokenized) {
    this.isTokenized = isTokenized;
    return this;
  }
  
  /**
   * Read Dataset from a given directory or file.
   */
  @Override
  public Dataset read(Resource path) throws IOException {
    if(path.isDirectory()) {
      return readDatasetFromDirectory(path, ".+");
    } else if(path.isFile()) {
      Document doc = readDocumentFromFile(path);
      Dataset data = new Dataset(path.getFileName());
      data.addDocument(doc);
      return data;
    } else throw new FileNotFoundException("cannot open path: " + path.toString());
  }
  
  /**
   * Read Dataset from a given directory of files.
   * @param pattern REGEX pattern to match only selected file names
   */
  public Dataset readDatasetFromDirectory(Resource path, String pattern) throws IOException {
    log.info("Reading Documents from {}", path.toString());
    Dataset data = new Dataset(path.getPath().getFileName().toString());
    AtomicInteger progress = new AtomicInteger();
    Stream<Path> paths = Files.walk(path.getPath())
        .filter(p -> Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS))
        .filter(p -> p.getFileName().toString().matches(pattern))
        .sorted();
    if(randomizeDocuments) {
      List<Path> list = paths.collect(Collectors.toList());
      Collections.shuffle(list);
      paths = list.stream();
    }
    Stream<Document> docs = paths
        .map(p -> readDocumentFromFile(Resource.fromFile(p.toString())))
        .filter(d -> !d.isEmpty());
    if(limit >= 0) {
      docs = docs.limit(limit);
    }
    docs.forEach(d -> {
      data.addDocument(d);
      int n = progress.incrementAndGet();
      if(n % 1000 == 0) {
        double free = Runtime.getRuntime().freeMemory() / (1024. * 1024. * 1024.);
        double total = Runtime.getRuntime().totalMemory() / (1024. * 1024. * 1024.);
        log.debug("read {}k documents, memory usage {} GB", n / 1000, (int)((total-free) * 10) / 10.);
      }
    });
    return data;
  }
  
  /**
   * Read a single Document from file.
   */
  public Document readDocumentFromFile(Resource file) {
    try(InputStream in = file.getInputStream()) {
        CharsetDecoder utf8 = StandardCharsets.UTF_8.newDecoder();
        BufferedReader br = new BufferedReader(new InputStreamReader(in, utf8));
        String text = br.lines().collect(Collectors.joining("\n"));
        Document doc = isTokenized ? 
            DocumentFactory.fromTokens(DocumentFactory.createTokensFromTokenizedText(text)) :
            DocumentFactory.fromText(text, DocumentFactory.Newlines.KEEP);
        doc.setId(file.getFileName());
        doc.setSource(file.toString());
        if(useFirstSentenceAsTitle) {
          if(doc.countSentences() > 0) {
            doc.setTitle(doc.getSentence(0).getText().trim());
          } else {
            doc.setTitle("");
          }
        }
        return doc;
    } catch(IOException ex) {
      // IOException is now allowed in Stream
      log.error(ex.toString());
      throw new RuntimeException(ex.toString(), ex.getCause());
    }
  }


}
