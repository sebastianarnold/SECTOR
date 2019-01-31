/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.datexis.preprocess;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import de.datexis.common.WordHelpers;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.tokenize.TokenContextGenerator;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * A Tokenizer for converting raw text into separated tokens (with newlines).
 * It uses Maximum Entropy to make its decisions.  The features are loosely
 * based off of Jeff Reynar's UPenn thesis "Topic Segmentation:
 * Algorithms and Applications.", which is available from his
 * homepage: <a href="http://www.cis.upenn.edu/~jcreynar">http://www.cis.upenn.edu/~jcreynar</a>.
 * @author additional Newline segmentation by Sebastian Arnold, Rudolf Schneider
 */
public class TokenizerMENL extends TokenizerME {

  /** fields inherited from TokenizerME through reflection */
  private Pattern alphanumeric;
  private MaxentModel model;
  private TokenContextGenerator cg;
  private List<Double> tokProbs;
  private List<Span> newTokens;
  
  public TokenizerMENL(TokenizerModel model) {
    super(model);
    initializeFieldsFromReflection();
  }

  private void initializeFieldsFromReflection() {
    try {
      // we need to get some fields from TokenizerME through reflection
      Field modelField = TokenizerME.class.getDeclaredField("model");
      modelField.setAccessible(true);
      this.model = (MaxentModel) modelField.get(this);
      Field tokProbsField = TokenizerME.class.getDeclaredField("tokProbs");
      tokProbsField.setAccessible(true);
      this.tokProbs = (List<Double>) tokProbsField.get(this);
      Field newTokensField = TokenizerME.class.getDeclaredField("newTokens");
      newTokensField.setAccessible(true);
      this.newTokens = (List<Span>) newTokensField.get(this);
      Field alphanumericField = TokenizerME.class.getDeclaredField("alphanumeric");
      alphanumericField.setAccessible(true);
      this.alphanumeric = (Pattern) alphanumericField.get(this);
      Field cgField = TokenizerME.class.getDeclaredField("cg");
      cgField.setAccessible(true);
      this.cg = (TokenContextGenerator) cgField.get(this);
    } catch(Exception ex) {
      Logger.getLogger(TokenizerMENL.class.getName()).log(Level.SEVERE, null, ex);
    }
  }
  
  @Override
  public Span[] tokenizePos(String s) {
    return tokenizePosWithNewline(s);
  }

  /**
   * Adapted from WhitespaceTokenizer
   */
  public Span[] tokenizePosWhitespaceWithNewline(String d) {
    int tokStart = -1;
    List<Span> tokens = new ArrayList<>();
    boolean inTok = false;

    //gather up potential tokens
    int end = d.length();
    for (int i = 0; i < end; i++) {
      char c = d.charAt(i);
      boolean isNewline = (c == '\n');
      boolean isSingleTokenChar = ("\"()[]{}".indexOf(c) != -1);
      if (StringUtil.isWhitespace(c) && !isNewline) {
        if (inTok) {
          // end token
          tokens.add(new Span(tokStart, i));
          inTok = false;
          tokStart = -1;
        }
      } else if(isNewline || isSingleTokenChar) {
        if (inTok) {
          // end token
          tokens.add(new Span(tokStart, i));
          inTok = false;
          tokStart = -1;
        }
        // add newline token
        tokens.add(new Span(i, i + 1));
      } else {
        if (!inTok) {
          tokStart = i;
          inTok = true;
        }
      }
    }

    if (inTok) {
      tokens.add(new Span(tokStart, end));
    }

    return tokens.toArray(new Span[tokens.size()]);
  }
  
  public Span[] tokenizePosWithNewline(String d) {
    Span[] tokens = tokenizePosWhitespaceWithNewline(d);
    newTokens.clear();
    tokProbs.clear();
    for (Span s : tokens) {
      String tok = d.substring(s.getStart(), s.getEnd());
      // Can't tokenize single characters
      if (tok.length() < 2) {
        newTokens.add(s);
        tokProbs.add(1d);
        // TODO: include abbreviation matches
      //} else if(tok.equals("e.g.")) {
      //  newTokens.add(s);
      //  tokProbs.add(1d);
      } else if (useAlphaNumericOptimization() && alphanumeric.matcher(tok).matches()) {
        newTokens.add(s);
        tokProbs.add(1d);
      } else if(WordHelpers.abbreviationsEN.contains(tok) || WordHelpers.abbreviationsDE.contains(tok)) {
        newTokens.add(s);
        tokProbs.add(1d);
      } else {
        int start = s.getStart();
        int end = s.getEnd();
        final int origStart = s.getStart();
        double tokenProb = 1.0;
        for (int j = origStart + 1; j < end; j++) {
          double[] probs =
              model.eval(cg.getContext(tok, j - origStart));
          String best = model.getBestOutcome(probs);
          tokenProb *= probs[model.getIndex(best)];
          if (best.equals(TokenizerME.SPLIT)) {
            newTokens.add(new Span(start, j));
            tokProbs.add(tokenProb);
            start = j;
            tokenProb = 1.0;
          }
        }
        newTokens.add(new Span(start, end));
        tokProbs.add(tokenProb);
      }
    }

    Span[] spans = new Span[newTokens.size()];
    newTokens.toArray(spans);
    return spans;
  }
  
}
