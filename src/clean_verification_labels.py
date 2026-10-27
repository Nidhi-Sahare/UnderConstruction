
"""
Clean visible verification wording across BuildProgress.

Changes presentation text only.
Does not modify API data, validation results, or verification logic.
"""

from flask import Response

SCRIPT = r"""
<script id="uc-clean-verification-labels">
(() => {
    "use strict";

    const replacements = [
        [/\bUNVERIFIED\b/g, "PENDING REVIEW"],
        [/\bUnverified\b/g, "Pending review"],
        [/\bunverified\b/g, "not yet confirmed"]
    ];

    function cleanText(text) {
        let result = text;

        for (const [pattern, replacement] of replacements) {
            result = result.replace(pattern, replacement);
        }

        return result;
    }

    function clean(root) {
        if (!root) return;

        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT
        );

        const nodes = [];

        while (walker.nextNode()) {
            const node = walker.currentNode;
            const parent = node.parentElement;

            if (!parent) continue;

            if (parent.closest(
                "script, style, textarea, code, pre, option"
            )) {
                continue;
            }

            if (/unverified/i.test(node.nodeValue)) {
                nodes.push(node);
            }
        }

        for (const node of nodes) {
            const updated = cleanText(node.nodeValue);

            if (updated !== node.nodeValue) {
                node.nodeValue = updated;
            }
        }

        document.querySelectorAll(
            "[title], [aria-label], [placeholder]"
        ).forEach(element => {
            for (const attribute of [
                "title", "aria-label", "placeholder"
            ]) {
                const value = element.getAttribute(attribute);

                if (value && /unverified/i.test(value)) {
                    element.setAttribute(
                        attribute,
                        cleanText(value)
                    );
                }
            }
        });
    }

    let scheduled = false;

    function scheduleClean() {
        if (scheduled) return;

        scheduled = true;

        requestAnimationFrame(() => {
            scheduled = false;
            clean(document.body);
        });
    }

    function initialize() {
        clean(document.body);

        const observer = new MutationObserver(scheduleClean);

        observer.observe(document.body, {
            childList: true,
            characterData: true,
            subtree: true
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            { once: true }
        );
    } else {
        initialize();
    }
})();
</script>
"""


def install_clean_verification_labels(app):
    @app.after_request
    def clean_verification_response(response):
        if response.direct_passthrough:
            return response

        if response.status_code != 200:
            return response

        if not response.mimetype == "text/html":
            return response

        html = response.get_data(as_text=True)

        if "uc-clean-verification-labels" in html:
            return response

        if "</body>" not in html.lower():
            return response

        import re

        html = re.sub(
            r"</body>",
            lambda match: SCRIPT + match.group(0),
            html,
            count=1,
            flags=re.IGNORECASE
        )

        response.set_data(html)
        response.headers["Cache-Control"] = "no-store"

        return response
