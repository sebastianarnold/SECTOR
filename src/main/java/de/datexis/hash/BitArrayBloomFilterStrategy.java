/*
 * Copyright (C) 2011 The Guava Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package de.datexis.hash;

import com.google.common.hash.Funnel;
import com.google.common.hash.Hashing;
import com.google.common.primitives.Longs;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * Collections of strategies of generating the k * log(M) bits required for an element to be mapped
 * to a BloomFilter of M bits and k hash functions. These strategies are part of the serialized form
 * of the Bloom filters that use them, thus they must be preserved as is (no updates allowed, only
 * introduction of new versions).
 *
 * Important: the order of the constants cannot change, and they cannot be deleted - we depend on
 * their ordinal for BloomFilter serialization.
 *
 * @author Dimitris Andreou
 * @author Kurt Alfred Kluever
 */
public class BitArrayBloomFilterStrategy implements BitArrayBloomFilter.Strategy {
  /**
   * This strategy uses all 128 bits of {@link Hashing#murmur3_128} when hashing. It looks different
   * than the implementation in MURMUR128_MITZ_32 because we're avoiding the multiplication in the
   * loop and doing a (much simpler) += hash2. We're also changing the index to a positive number by
   * AND'ing with Long.MAX_VALUE instead of flipping the bits.
   */
  //MURMUR128_MITZ_64() {
    @Override
    public <T> boolean put(
      T object, Funnel<? super T> funnel, int numHashFunctions, BitArrayBloomFilter.BitArray bits) {
      long bitSize = bits.bitSize();
      boolean bitsChanged = false;
      try { // we need to access the method using reflection
        Object obj = Hashing.murmur3_128().hashObject(object, funnel);
        Method method = obj.getClass().getDeclaredMethod("getBytesInternal");
        method.setAccessible(true);
        byte[] bytes = (byte[]) method.invoke(obj);
        long hash1 = lowerEight(bytes);
        long hash2 = upperEight(bytes);
        long combinedHash = hash1;
        for (int i = 0; i < numHashFunctions; i++) {
          // Make the combined hash positive and indexable
          bitsChanged |= bits.set((combinedHash & Long.MAX_VALUE) % bitSize);
          combinedHash += hash2;
        }
      } catch(NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
        e.printStackTrace();
      }
      return bitsChanged;
    }

    @Override
    public <T> boolean mightContain(
        T object, Funnel<? super T> funnel, int numHashFunctions, BitArrayBloomFilter.BitArray bits) {
      long bitSize = bits.bitSize();
      try { // we need to access the method using reflection
        Object obj = Hashing.murmur3_128().hashObject(object, funnel);
        Method method = obj.getClass().getDeclaredMethod("getBytesInternal");
        method.setAccessible(true);
        byte[] bytes = (byte[]) method.invoke(obj);
        long hash1 = lowerEight(bytes);
        long hash2 = upperEight(bytes);
        long combinedHash = hash1;
        for (int i = 0; i < numHashFunctions; i++) {
          // Make the combined hash positive and indexable
          if (!bits.get((combinedHash & Long.MAX_VALUE) % bitSize)) {
            return false;
          }
          combinedHash += hash2;
        }
      } catch(NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
        e.printStackTrace();
      }
      return true;
    }

    public <T> double[] getBitArray(T object, Funnel<? super T> funnel,
        int numHashFunctions, BitArrayBloomFilter.BitArray bits) {
      long bitSize = bits.bitSize();
      double[] arr = new double[(int)bitSize];
      try { // we need to access the method using reflection
        Object obj = Hashing.murmur3_128().hashObject(object, funnel);
        Method method = obj.getClass().getDeclaredMethod("getBytesInternal");
        method.setAccessible(true);
        byte[] bytes = (byte[]) method.invoke(obj);
        long hash1 = lowerEight(bytes);
        long hash2 = upperEight(bytes);

        long combinedHash = hash1;
        for (int i = 0; i < numHashFunctions; i++) {
          // Make the combined hash positive and indexable
          int idx = (int) ((combinedHash & Long.MAX_VALUE) % bitSize);
          arr[idx] = bits.get(idx) ? 1.0 : 0.0;
          combinedHash += hash2;
        }
      } catch(NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
        e.printStackTrace();
      }
      return arr;
    }

    private /* static */ long lowerEight(byte[] bytes) {
      return Longs.fromBytes(
          bytes[7], bytes[6], bytes[5], bytes[4], bytes[3], bytes[2], bytes[1], bytes[0]);
    }

    private /* static */ long upperEight(byte[] bytes) {
      return Longs.fromBytes(
          bytes[15], bytes[14], bytes[13], bytes[12], bytes[11], bytes[10], bytes[9], bytes[8]);
    }

  @Override
  public int ordinal() {
    return 3;
  }
  
}