review_patterns = [
    {'tag': 'span', 'attrs': {'data-hook': 'review-body'}},
    {'tag': 'div', 'attrs': {'class': 'review-text'}},
    {'tag': 'p', 'attrs': {'itemprop': 'reviewBody'}},
    {'tag': 'div', 'attrs': {'class': 'comments-content'}},
    {'tag': 'span', 'attrs': {'class': 'feedback-item-content'}},
    {'tag': 'div', 'attrs': {'class': 'voice-comment__text'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-item'}},
    {'tag': 'div', 'attrs': {'class': 'customer-review-text'}},
    {'tag': 'span', 'attrs': {'class': 'review-body-text'}},
    {'tag': 'div', 'attrs': {'class': 'review-comment-body'}},
    {'tag': 'p', 'attrs': {'class': 'review-comment-text'}},
    {'tag': 'div', 'attrs': {'class': 'user-review'}},
    {'tag': 'div', 'attrs': {'class': 'product-review-content'}},
    {'tag': 'span', 'attrs': {'class': 'product-review-text'}},
    {'tag': 'div', 'attrs': {'class': 'ugc-review'}},
    {'tag': 'p', 'attrs': {'class': 'ugc-review-text'}},
    {'tag': 'div', 'attrs': {'class': 'customer-review-body'}},
    {'tag': 'p', 'attrs': {'class': 'customer-review-text'}},
    {'tag': 'div', 'attrs': {'class': 'comment-body'}},
    {'tag': 'p', 'attrs': {'class': 'comment-text'}},
    {'tag': 'span', 'attrs': {'class': 'review-content'}},
    {'tag': 'div', 'attrs': {'class': 'product-review-text'}},
    {'tag': 'p', 'attrs': {'class': 'review-text'}},
    {'tag': 'div', 'attrs': {'class': 'review-comment'}},
    {'tag': 'p', 'attrs': {'class': 'comment-description'}},
    {'tag': 'div', 'attrs': {'class': 'review'}},
    {'tag': 'span', 'attrs': {'class': 'text-review'}},
    {'tag': 'div', 'attrs': {'class': 'review-description'}},
    {'tag': 'p', 'attrs': {'class': 'review-details'}},
    {'tag': 'div', 'attrs': {'class': 'review-content-wrapper'}},
    {'tag': 'p', 'attrs': {'class': 'review-body-content'}},
    {'tag': 'div', 'attrs': {'class': 'feedback'}},
    {'tag': 'span', 'attrs': {'class': 'user-feedback'}},
    {'tag': 'div', 'attrs': {'class': 'review-message'}},
    {'tag': 'p', 'attrs': {'class': 'feedback-text'}},
    {'tag': 'div', 'attrs': {'class': 'review-entry'}},
    {'tag': 'div', 'attrs': {'class': 'user-review-body'}},
    {'tag': 'p', 'attrs': {'class': 'feedback-body'}},
    {'tag': 'div', 'attrs': {'class': 'review-summary'}},
    {'tag': 'span', 'attrs': {'class': 'comment-content'}},
    {'tag': 'div', 'attrs': {'class': 'review-feedback'}},
    {'tag': 'p', 'attrs': {'class': 'feedback-comment'}},
    {'tag': 'div', 'attrs': {'class': 'customer-opinion'}},
    {'tag': 'p', 'attrs': {'class': 'review-feedback-text'}},
    {'tag': 'span', 'attrs': {'class': 'customer-comment'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-review-text'}},
    {'tag': 'p', 'attrs': {'class': 'opinion-text'}},
    {'tag': 'div', 'attrs': {'class': 'product-comment'}},
    {'tag': 'p', 'attrs': {'class': 'product-comment-text'}},
    {'tag': 'div', 'attrs': {'class': 'review-opinion'}},
    {'tag': 'span', 'attrs': {'class': 'customer-feedback'}},
    {'tag': 'div', 'attrs': {'class': 'product-review-body'}},
    {'tag': 'p', 'attrs': {'class': 'product-feedback'}},
    {'tag': 'span', 'attrs': {'class': 'product-review-description'}},
    {'tag': 'div', 'attrs': {'class': 'comment-text'}},
    {'tag': 'p', 'attrs': {'class': 'review-description-text'}},
    {'tag': 'div', 'attrs': {'class': 'ugc-comment'}},
    {'tag': 'span', 'attrs': {'class': 'ugc-comment-text'}},
    {'tag': 'div', 'attrs': {'class': 'customer-feedback-body'}},
    {'tag': 'p', 'attrs': {'class': 'review-message-text'}},
    {'tag': 'span', 'attrs': {'class': 'review-feedback-body'}},
    {'tag': 'div', 'attrs': {'class': 'review-user-body'}},
    {'tag': 'p', 'attrs': {'class': 'review-opinion-text'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-comment-body'}},
    {'tag': 'span', 'attrs': {'class': 'review-feedback-content'}},
    {'tag': 'div', 'attrs': {'class': 'customer-review-summary'}},
    {'tag': 'p', 'attrs': {'class': 'review-summary-text'}},
    {'tag': 'div', 'attrs': {'class': 'user-review-content'}},
    {'tag': 'span', 'attrs': {'class': 'review-body-comment'}},
    {'tag': 'div', 'attrs': {'class': 'user-review-comment'}},
    {'tag': 'p', 'attrs': {'class': 'comment-feedback'}},
    {'tag': 'span', 'attrs': {'class': 'review-feedback-summary'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-description'}},
    {'tag': 'p', 'attrs': {'class': 'comment-summary'}},
    {'tag': 'span', 'attrs': {'class': 'user-review-summary'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-body-content'}},
    {'tag': 'p', 'attrs': {'class': 'product-feedback-comment'}},
    {'tag': 'span', 'attrs': {'class': 'review-body-feedback'}},
    {'tag': 'div', 'attrs': {'class': 'comment-feedback-body'}},
    {'tag': 'p', 'attrs': {'class': 'product-review-comment'}},
    {'tag': 'span', 'attrs': {'class': 'review-text-feedback'}},
    {'tag': 'div', 'attrs': {'class': 'review-feedback-summary-body'}},
    {'tag': 'p', 'attrs': {'class': 'customer-opinion-feedback'}},
    {'tag': 'span', 'attrs': {'class': 'product-feedback-summary'}},
    {'tag': 'div', 'attrs': {'class': 'feedback-opinion-body'}},
    {'tag': 'p', 'attrs': {'class': 'user-feedback-comment'}},
    {'tag': 'span', 'attrs': {'class': 'review-opinion-summary'}},
    {'tag': 'div', 'attrs': {'class': 'product-opinion-summary'}},
    {'tag': 'p', 'attrs': {'class': 'comment-review-body'}},
    {'tag': 'span', 'attrs': {'class': 'review-comment-summary'}},
    {'tag': 'div', 'attrs': {'class': 'ugc-feedback-comment'}},
    {'tag': 'p', 'attrs': {'class': 'feedback-review-summary'}},
    {'tag': 'span', 'attrs': {'class': 'customer-review-comment'}},
    {'tag': 'div', 'attrs': {'class': 'customer-opinion-summary'}},
    {'tag': 'p', 'attrs': {'class': 'feedback-review-body'}}
]