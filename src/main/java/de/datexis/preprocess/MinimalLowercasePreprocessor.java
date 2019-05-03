package de.datexis.preprocess;

import de.datexis.common.WordHelpers;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

/**
 * Proprocessor used for single tokens without positional information.
 * Converts to lowercase and strips punctuation and numbers.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MinimalLowercasePreprocessor implements TokenPreProcess {
  @Override
  public String preProcess(String token) {
    if(token == null) return null;
    token = token.trim();
    token = WordHelpers.replaceUmlauts(token);
    token = WordHelpers.replaceAccents(token);
    token = WordHelpers.replacePunctuation(token, "");
    token = WordHelpers.replaceNumbers(token, "#");
    token = WordHelpers.replaceSpaces(token, "_");
    return token.toLowerCase();
  }
}
