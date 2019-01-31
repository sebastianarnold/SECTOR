package de.datexis.common;

import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Iterator;
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.reflections.Reflections;
import org.reflections.util.ConfigurationBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class for Object JSON serialization
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ObjectSerializer {

  protected final static Logger log = LoggerFactory.getLogger(ObjectSerializer.class);

  private static ObjectMapper mapper = null;
  
  private static ObjectMapper getInstance() {
    if(mapper == null) {
      mapper = new ObjectMapper();
      org.reflections.Configuration conf = ConfigurationBuilder.build("de.datexis").setExpandSuperTypes(false);
      Reflections reflections = new Reflections(conf);
      for(Class<? extends Annotation> c : reflections.getSubTypesOf(Annotation.class)) mapper.registerSubtypes(c);
      mapper.setSerializationInclusion(Include.NON_NULL);
    }
    return mapper;
  }
 
  public static String getJSON(Object o) {
    try {
      return getInstance().writerWithDefaultPrettyPrinter().writeValueAsString(o);
    } catch (JsonProcessingException ex) {
      log.error("Error generating model JSON: " + ex.toString());
      return null;
    }
  }
  
  public static String getJSONRaw(Object o) {
    try {
      return getInstance().writer().writeValueAsString(o);
    } catch (JsonProcessingException ex) {
      log.error("Error generating model JSON: " + ex.toString());
      return null;
    }
  }
  
  public static void writeJSON(Object o, Resource res) {
    try {
      getInstance().writerWithDefaultPrettyPrinter().writeValue(res.getOutputStream(), o);
    } catch (JsonProcessingException ex) {
      log.error("Error saving model JSON: " + ex.toString());
    } catch (IOException ex) {
      log.error("Error saving model JSON: " + ex.toString());
    }
  }
  
  public static void writeJSONRaw(Object o, Resource res) {
    try {
      getInstance().writer().writeValue(res.getOutputStream(), o);
    } catch (JsonProcessingException ex) {
      log.error("Error saving model JSON: " + ex.toString());
    } catch (IOException ex) {
      log.error("Error saving model JSON: " + ex.toString());
    }
  }
  
  public static Iterator<Document> readJSONDocumentIterable(Resource res) throws IOException {
    return getInstance().readerFor(Document.class).readValues(res.getInputStream());
  }
  
  public static ObjectMapper getObjectMapper() {
    return getInstance();
  }
  
  public static <T extends Object> T readFromJSON(String json, Class<T> type) throws IOException {
    return getInstance().readerFor(type).readValue(json);
  }
  
  public static <T extends Object> T readFromJSON(Resource res, Class<T> type) throws IOException {
    String json = IOUtils.toString(res.getInputStream(), "UTF-8");
    //return getInstance().readerFor(type).readValue(res.getInputStream()); // does not work for multi-line JSON
    return getInstance().readerFor(type).readValue(json);
  }
 
  public static String getArrayAsBase64String(INDArray arr) {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    BufferedOutputStream bos = new BufferedOutputStream(baos);
    try(DataOutputStream dos = new DataOutputStream(bos)) {
      Nd4j.write(arr, dos);
      dos.flush();
      byte[] encodedBytes = Base64.encodeBase64(baos.toByteArray());
      return new String(encodedBytes);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not encode INDArray as Base64");
    }
  }
  
  public static INDArray getArrayFromBase64String(String encoded) {
    byte[] decodedBytes = Base64.decodeBase64(encoded);
    ByteArrayInputStream bais = new ByteArrayInputStream(decodedBytes);
    BufferedInputStream bis = new BufferedInputStream(bais);
    try(DataInputStream dis = new DataInputStream(bis)) {
      return Nd4j.read(dis);
    } catch(IOException ex) {
      throw new RuntimeException("Could not create INDArray from Base64 String");
    }
  }
  
  
}
