package de.datexis.common;

import org.apache.commons.cli.*;

/**
 * The CommandLineParser parse the args from the command line and tries to map
 * them to the given strategy.
 *
 * @author Robert Dziuba s58345@beuth-hochschule.de
 */
public class CommandLineParser {

  private final Options options;

  /**
   * Instance og a EntityLinkerParamsParser.
   *
   * @param options on which the args should be mapped
   */
  public CommandLineParser(Options options) {
    this.options = options;
  }

  /**
   * Parse the command line args and maps it on the given strategy.
   *
   * @param args command line line args
   * @throws ParseException if the mapping goes wrong
   */
  public void parse(String[] args) throws ParseException {
    org.apache.commons.cli.Options opts = options.setUpCliOptions();
    org.apache.commons.cli.CommandLineParser defaultParser = new DefaultParser();
    CommandLine parse = defaultParser.parse(opts, args);
    options.setParams(parse);
  }

  /**
   * Print help.
   * 
   * @param cmd teh command to use
   */
  public void printHelp(String cmd) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp(cmd, options.setUpCliOptions());
  }
  
  /**
   * Interface for the comand line parameter Strategy.
   *
   * @author Robert Dziuba s58345@beuth-hochschule.de
   */
  public interface Options {

    /**
     * Returns the apache commons cli options with the required comand line
     * parameter
     *
     * @return apache commons cli options
     */
    org.apache.commons.cli.Options setUpCliOptions();

    /**
     * Sets the parsed command line parameter.
     *
     * @param parse
     */
    void setParams(CommandLine parse);
  }

}
