package de.datexis.common;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.slf4j.LoggerFactory;

/**
 * Read resources from the local file system
 * @author sarnold
 */
public class ExternalResource extends Resource {

  protected static final org.slf4j.Logger log = LoggerFactory.getLogger(ExternalResource.class);
  
  public ExternalResource(String path) {
    this.path = path.replace("\\", "/");
  }

  /**
   * Returns a resource from external path as stream. If the file is GZIPped, the stream is converted.
   * @return empty Stream on File not found
   */
  @Override
  public InputStream getInputStream() throws IOException {
    InputStream in = null;
    try {
      in = new BufferedInputStream(new FileInputStream(path));
      // TODO: checking file name is not optimal
      if(getFileName().endsWith(".gz")) {
        InputStream gzip = new GZIPInputStream(in);
        return gzip;
      } else if(getFileName().endsWith(".bz2")) {
        InputStream bz2 = new BZip2CompressorInputStream(in, true);
        return bz2;
      }
      /*try (BufferedInputStream bis = new BufferedInputStream(
                GzipUtils.isCompressedFilename(modelFile.getName())
                        ? new GZIPInputStream(new FileInputStream(modelFile))
                        : new FileInputStream(modelFile));
          DataInputStream dis = new DataInputStream(bis)) {*/
      return in;
    } catch(ZipException e) {
      return in;
    }
  }
  
  /**
   * Writes into an external resource
   * @return empty Stream on File not found
   */
  @Override
  public OutputStream getOutputStream(boolean append) throws IOException {
    return new FileOutputStream(toFile(), append);
  }

  @Override
  public File toFile() {
    return new File(path);
  }

  @Override
  public Path getPath() {
    return Paths.get(path);
  }
  
  @Override
  public String getFileName() {
		return getPath().getFileName().toString();
  }
  
  @Override
  public Resource resolve(String path) {
    return new ExternalResource(this.path.replaceFirst("/$", "") + "/" + path.replaceFirst("^/", ""));
  }

  @Override
  public boolean exists() {
    return toFile().exists();
  }

  @Override
  public boolean isFile() {
    return toFile().isFile();
  }

  @Override
  public boolean isDirectory() {
    return toFile().isDirectory();
  }
  
 }
