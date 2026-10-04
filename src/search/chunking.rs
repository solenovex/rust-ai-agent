pub fn fixed_length_chunking(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    assert!(chunk_size > 0, "chunk_size must be greater than 0");
    assert!(overlap < chunk_size, "overlap must be less than chunk_size");

    let chars: Vec<char> = text.chars().collect();
    let chars_len = chars.len();
    let mut chunks = Vec::new();
    let step = chunk_size - overlap;

    let mut start = 0usize;
    while start < chars_len {
        let end = (start + chunk_size).min(chars_len);
        chunks.push(chars[start..end].iter().collect());
        if end == chars_len {
            break;
        }
        start += step;
    }

    chunks
}

#[cfg(test)]
mod chunking_tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let result = fixed_length_chunking("abcdefghij", 5, 0);

        assert_eq!(result, vec!["abcde".to_string(), "fghij".to_string(),]);
    }

    #[test]
    fn test_chunking_with_overlap() {
        let result = fixed_length_chunking("abcdefghij", 5, 2);

        assert_eq!(
            result,
            vec!["abcde".to_string(), "defgh".to_string(), "ghij".to_string(),]
        );
    }

    #[test]
    fn test_final_chunk_is_shorter() {
        let result = fixed_length_chunking("abcdefgh", 5, 0);

        assert_eq!(result, vec!["abcde".to_string(), "fgh".to_string(),]);
    }

    #[test]
    fn test_chunk_size_larger_than_text() {
        let result = fixed_length_chunking("abc", 10, 0);

        assert_eq!(result, vec!["abc".to_string()]);
    }

    #[test]
    fn test_empty_text() {
        let result = fixed_length_chunking("", 5, 2);

        assert!(result.is_empty());
    }

    #[test]
    fn test_unicode_text() {
        let result = fixed_length_chunking("你好世界朋友", 3, 1);

        assert_eq!(
            result,
            vec![
                "你好世".to_string(),
                "世界朋".to_string(),
                "朋友".to_string(),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than 0")]
    fn test_zero_chunk_size() {
        fixed_length_chunking("abc", 0, 0);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than chunk_size")]
    fn test_overlap_equal_to_chunk_size() {
        fixed_length_chunking("abc", 3, 3);
    }

    #[test]
    #[should_panic(expected = "overlap must be less than chunk_size")]
    fn test_overlap_greater_than_chunk_size() {
        fixed_length_chunking("abc", 3, 4);
    }
}
