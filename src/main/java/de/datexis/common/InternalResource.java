package de.datexis.common;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.slf4j.LoggerFactory;

/**
 * Read resources from JAR (no File access possible)
 * @author sarnold
 */
public class InternalResource extends Resource {

  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(InternalResource.class);
  
  public InternalResource(String path) {
    this.path = path;
    // TODO: use ClassPathResource!
  }

  /**
   * Returns a resource from JAR or classpath as stream
   * @return empty Stream on File not found
   */
  @Override
  public InputStream getInputStream() {
    InputStream in = null;
    try {
      in = Configuration.class.getClassLoader().getResourceAsStream(path);
      if(in == null) throw new RuntimeException("Could not read JAR resource: \"" + path + "\"");
      // TODO: checking file name is not optimal
      if(getFileName().endsWith(".gz")) {
        InputStream gzip = new GZIPInputStream(in);
        return gzip;
      } else if (this.getFileName().endsWith(".bz2")) {
        InputStream bz2 = new BZip2CompressorInputStream(in, true);
        return bz2;
      }
      return in;
    } catch(ZipException e) {
      return in;
    } catch (FileNotFoundException ex) {
      throw new RuntimeException("JAR resource not found: \"" + path + "\"");
    } catch (IOException ex) {
      throw new RuntimeException("Could not read JAR resource: \"" + path + "\"");
    }
  }
  
  /**
   * Writes into an external resource
   * @return empty Stream on File not found
   */
  @Override
  public OutputStream getOutputStream(boolean append) {
    throw new UnsupportedOperationException("You cannot write into an internal resource.");
  }

  @Override
  public File toFile() {
    Resource temp = createTempFile(getFileName());
    try(InputStream in = getInputStream()) {
      Files.copy(in, temp.getPath(), StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException ex) {
      throw new UnsupportedOperationException("Could not copy Resource file to Temp.");
    }
    return temp.toFile();
  }
  
  @Override
  public Path getPath() {
    return toFile().toPath();
  }

  @Override
  public String getFileName() {
		return path.substring(path.lastIndexOf("/")+1);
  }
  
  @Override
  public Resource resolve(String path) {
    return new InternalResource(this.path.replaceFirst("/$", "") + "/" + path.replaceFirst("^/", ""));
  }
  
  @Override
  public boolean exists() {
    URL u = Configuration.class.getClassLoader().getResource(path);
    return u != null;
  }

  @Override
  public boolean isFile() {
    //URL u = Configuration.class.getClassLoader().getResource(path);
    //return u != null && u.getProtocol().equals("file");
    throw new UnsupportedOperationException("cannot access JAR resource as directory");
  }

  @Override
  public boolean isDirectory() {
    throw new UnsupportedOperationException("cannot access JAR resource as directory");
  }

}
