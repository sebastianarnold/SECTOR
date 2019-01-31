package de.datexis.common;

import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;
import org.slf4j.LoggerFactory;

/**
 * Interface to access File and Path resources
 * @author sarnold
 */
public abstract class Resource {
  
  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(Resource.class);

  protected String path;
  
  public abstract Resource resolve(String path);
  public abstract InputStream getInputStream() throws IOException;
  public abstract OutputStream getOutputStream(boolean append) throws IOException;
  public abstract File toFile();
  public abstract Path getPath();
  
  public OutputStream getOutputStream() throws IOException {
    return getOutputStream(false);
  }
  
  /**
   * @return file name of the Resource (without path)
   */
  public abstract String getFileName();
  
  public GZIPOutputStream getGZIPOutputStream() throws IOException {
    return new GZIPOutputStream(getOutputStream());
  }
  
  @Override
  public String toString() {
    return path;
  }
  
  public abstract boolean exists();
  
  public abstract boolean isFile();
  
  public abstract boolean isDirectory();
  
  /**
   * Creates access to an internal file from JAR or resources folder.
   * @param file
   * @return 
   */
  public static Resource fromJAR(String file) {
    return new InternalResource(file);
  }
  
  /**
   * Creates access to an external file from file system.
   * @param path
   * @param file
   * @return 
   */
  public static Resource fromFile(String path, String file) {
    path = path.replaceFirst("^~", getUserHome());
    return new ExternalResource(path + "/" + file);
  }
  
  public static Resource fromFile(String path) {
    path = path.replaceFirst("^~", getUserHome());
    return new ExternalResource(path);
  }

  /**
   * Creates access to an external directory from file system path.
   * @param path
   * @return 
   */
  public static Resource fromDirectory(Path path) {
    return new ExternalResource(path.toString());
  }
  
  /**
   * Creates access to an external directory from file system path.
   * @param path
   * @return 
   */
  public static Resource fromDirectory(String path) {
    path = path.replaceFirst("^~", getUserHome());
    return new ExternalResource(path);
  }
  
  /**
   * Creates access to a temporary file
   * @param name
   * @return 
   */
  public static Resource createTempFile(String name) {
    try {
      File temp = File.createTempFile(name, ".tmp");
      temp.deleteOnExit();
      return new ExternalResource(temp.getAbsolutePath());
    } catch (IOException ex) {
      log.error("Could not create temp file: " + ex.toString());
      return null;
    }
  }
  
  /**
   * Creates access to a temporary directory
   * @return 
   */
  public static Resource createTempDirectory() {
    try {
      File temp = Files.createTempDir();
      temp.deleteOnExit();
      return new ExternalResource(temp.getAbsolutePath());
    } catch (Exception ex) {
      log.error("Could not create temp dir: " + ex.toString());
      return null;
    }
  }
  
  /**
   * Creates access to a file or directory that is configured in properties file.
   * @param key relative pathnames are loaded from JAR, absolute pathnames starting with /, X:, ./ or ~/ are loaded from file system.
   * @return 
   */
  public static Resource fromConfig(String key) {
    String path = Configuration.getProperty(key);
    path = path.replaceFirst("^~", getUserHome());
    if(path.matches("^([a-zA-Z]:)?\\.?[/\\\\].*")) return new ExternalResource(path);
    else return new InternalResource(path);
  }

  public static String getUserHome() {
    return System.getProperty("user.home").replace('\\', '/');
  }
  
  public static Resource fromConfig(String key, String file) {
    return fromFile(Configuration.getProperty(key), file);
  }

  public static Resource fromURL(String url) {
    // TODO: implement: REST call
    // TODO: implement: Wget download
    // TODO: implement: Scp copy?
    throw new UnsupportedOperationException();
  }
  
}