package de.datexis.encoder.impl;

import de.datexis.hash.BitArrayBloomFilter;
import de.datexis.hash.BitArrayBloomFilterStrategy;
import com.google.common.hash.Funnels;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.model.Document;
import de.datexis.model.Span;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Collection;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * A Stub for Bloom Filter Encoder on top of Bag Of Words
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class BloomEncoder extends BagOfWordsEncoder {

  protected BitArrayBloomFilter<CharSequence> bloom;
  
  public BloomEncoder() {
    this("BLM", 4096, 5);
  }
  
  public BloomEncoder(int bitSize, int hashFunctions) {
    this("BLM", bitSize, hashFunctions);
  }
  
  public BloomEncoder(String id, int bitSize, int hashFunctions) {
    super(id);
    log = LoggerFactory.getLogger(BloomEncoder.class);
    bloom = BitArrayBloomFilter.create(Funnels.stringFunnel(Charset.defaultCharset()), bitSize, hashFunctions, new BitArrayBloomFilterStrategy());
  }
  
  @Override
  public String getName() {
    return "Bloom Filter Encoder";
  }

  @Override
  public void trainModel(Collection<Document> documents) {
    trainModel(documents, 1, WordHelpers.Language.EN);
  }
  
  @Override
  public void trainModel(Collection<Document> documents, int minWordFrequency, WordHelpers.Language language) {
    super.trainModel(documents, minWordFrequency, language);
    for(String word : getWords()) {
      bloom.put(word);
    }
    appendTrainLog("trained Bloom filter over " + vocab.numWords() + " words into " + bloom.bitSize() + " bits (ratio: " + ((double) bloom.bitSize() / vocab.numWords()));
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return bloom.bitSize();
  }
  
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    for(Span s : spans) {
      double[] bits = bloom.getBitArray(preprocessor.preProcess(s.getText()));
      INDArray x = Nd4j.create(bits);
      vector.addi(x.transposei());
    }
    return vector;
  }
  
  @Override
  public INDArray encode(String[] words) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    for(String s : words) {
      double[] bits = bloom.getBitArray(preprocessor.preProcess(s));
      INDArray x = Nd4j.create(bits);
      vector.addi(x.transposei());
    }
    return vector;
  }
  
  private static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
    byte[] bytes = new byte[1024];
    int bytesRead;
    while ((bytesRead = inputStream.read(bytes)) != -1) {
      zipStream.write(bytes, 0, bytesRead);
    }
  }
  
  @Override
  public void saveModel(Resource modelPath, String name) {
    Resource modelFile = modelPath.resolve(name + ".zip");
    try(OutputStream out = modelFile.getOutputStream()) {
      
      Resource temp = Resource.createTempDirectory();
      ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(out)));
      
      // write vocab
      zipfile.putNextEntry(new ZipEntry("vocab.tsv"));
      super.saveModel(temp, "vocab");

      BufferedInputStream fis = new BufferedInputStream(temp.resolve("vocab.tsv.gz").getInputStream());
      writeEntry(fis, zipfile);
      fis.close();
        
      // write bloom filter
      zipfile.putNextEntry(new ZipEntry("bloom.bin"));
      OutputStream blm = temp.resolve("bloom.bin").getOutputStream();
      bloom.writeTo(blm);
      blm.flush();
      blm.close();
      
      fis = new BufferedInputStream(temp.resolve("bloom.bin").getInputStream());
      writeEntry(fis, zipfile);
      fis.close();
      
      zipfile.flush();
      zipfile.close();
      
      setModel(modelFile);
      setModelAvailable(true);
      log.info("saved bloom filter");
      
    } catch(IOException ex) {
      log.error(ex.toString());
    }
      
  }
  
  @Override
  public void loadModel(Resource modelFile) {
    try {
      
      Resource temp = Resource.createTempDirectory();
      ZipFile zipFile = new ZipFile(modelFile.toFile());
      
      // read vocab
      InputStream stream = zipFile.getInputStream(zipFile.getEntry("vocab.tsv"));
      Files.copy(stream, temp.resolve("vocab.tsv").getPath(), StandardCopyOption.REPLACE_EXISTING);
      super.loadModel(temp.resolve("vocab.tsv"));

      // read bloom filter
      stream = zipFile.getInputStream(zipFile.getEntry("bloom.bin"));
      Files.copy(stream, temp.resolve("bloom.bin").getPath(), StandardCopyOption.REPLACE_EXISTING);
      bloom = BitArrayBloomFilter.readFrom(temp.resolve("bloom.bin").getInputStream(), 
          Funnels.stringFunnel(Charset.defaultCharset()),
          new BitArrayBloomFilterStrategy());
      
      setModel(modelFile);
      setModelAvailable(true);
      log.info("loaded bloom filter with size " + getEmbeddingVectorSize());
      
    } catch(IOException ex) {
      log.error(ex.toString());
    }
  }
  
}
