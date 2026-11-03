// ---- Image mode ----

let imageReferenceFiles = [];

function targetSupportsImageOperation(target, operation) {
    return target?.type === 'copilot' && (target.supportedEndpoints || []).some(
        e => e === '/images/' + operation || e === '/v1/images/' + operation
    );
}

function targetSupportsImages(target) {
    return target?.type === 'copilot' && (target.modelType === 'image' ||
        targetSupportsImageOperation(target, 'generations') || targetSupportsImageOperation(target, 'edits'));
}

function selectImageFiles(files) {
    imageReferenceFiles = Array.from(files);
    if (!imageReferenceFiles.length) document.getElementById('image-files').value = '';
    document.getElementById('image-file-label').textContent = imageReferenceFiles.map(f => f.name).join(', ');
    if (imageReferenceFiles.length) document.getElementById('image-operation').value = 'edits';
}

// Keep binary-sized responses out of the session list's small localStorage quota.
// A storage failure must never hide an already-generated, billable image.
function imageStore(action, key, value) {
    return new Promise((resolve, reject) => {
        const opening = indexedDB.open('ghc-chat-images', 1);
        opening.onupgradeneeded = () => opening.result.createObjectStore('images');
        opening.onerror = () => reject(opening.error);
        opening.onblocked = () => reject(new Error('Image storage is blocked by another tab.'));
        opening.onsuccess = () => {
            const db = opening.result;
            const tx = db.transaction('images', action === 'get' ? 'readonly' : 'readwrite');
            const store = tx.objectStore('images');
            const result = action === 'put' ? store.put(value, key) :
                action === 'delete' ? store.delete(key) : store.get(key);
            tx.oncomplete = () => { db.close(); resolve(result.result); };
            tx.onabort = () => { db.close(); reject(tx.error || new Error('Image storage failed.')); };
        };
    });
}

function imageRequestSummary(payload) {
    return JSON.stringify(payload, (key, value) =>
        typeof value === 'string' && (key === 'b64_json' || value.startsWith('data:image/') || value.length > 16000)
            ? `[image/large value omitted: ${value.length} characters]` : value, 2);
}

function imageMessageElement(role) {
    const container = document.getElementById('messages-container');
    const empty = container.querySelector('.empty-state');
    if (empty) empty.remove();
    const div = document.createElement('div');
    div.className = 'message ' + role;
    container.appendChild(div);
    return div;
}

function imageText(parent, text, className = '') {
    const div = document.createElement('div');
    div.className = className;
    div.textContent = text;
    parent.appendChild(div);
    return div;
}

function appendImageRequest(content, endpoint) {
    const div = imageMessageElement('image-request');
    imageText(div, endpoint, 'image-label');
    imageText(div, content);
    return div;
}

function renderImageResponse(div, result, status, duration) {
    div.innerHTML = '';
    const usage = result?.usage;
    imageText(div, `Image ${status || ''}${duration != null ? ' · ' + formatDuration(duration) : ''}` +
        (usage ? ` · tokens ${usage.input_tokens || 0} in / ${usage.output_tokens || 0} out` : ''), 'image-label');
    if (status >= 400 || !Array.isArray(result?.data)) {
        const error = result?.error?.message || result?.detail || result?.error || result;
        imageText(div, typeof error === 'string' ? error : JSON.stringify(error), 'image-error');
        return;
    }
    const format = ['png', 'jpeg', 'webp'].includes(result.output_format) ? result.output_format : 'png';
    const gallery = document.createElement('div');
    gallery.className = 'image-gallery';
    div.appendChild(gallery);
    let rendered = 0;
    for (const [index, item] of result.data.entries()) {
        // Never allow arbitrary data: (e.g. SVG/HTML) or javascript: URLs.
        const source = typeof item.b64_json === 'string' && /^[A-Za-z0-9+/=\r\n]+$/.test(item.b64_json)
            ? `data:image/${format};base64,${item.b64_json}` :
                (typeof item.url === 'string' && item.url.trim() ? safeHttpUrl(item.url) : null);
        if (!source) continue;
        rendered++;
        const figure = document.createElement('figure');
        const image = document.createElement('img');
        image.src = source;
        image.alt = item.revised_prompt || 'Generated image ' + (index + 1);
        figure.appendChild(image);
        const download = document.createElement('a');
        download.href = source;
        download.download = `image-${index + 1}.${format}`;
        download.textContent = 'Download original';
        download.target = '_blank';
        download.rel = 'noopener noreferrer';
        figure.appendChild(download);
        if (item.b64_json) {
            const edit = document.createElement('button');
            edit.textContent = 'Use as reference';
            edit.onclick = () => {
                const bytes = Uint8Array.from(atob(item.b64_json), c => c.charCodeAt(0));
                selectImageFiles([new File([bytes], `reference.${format}`, {type: 'image/' + format})]);
                setMode('image');
            };
            figure.appendChild(edit);
        }
        if (item.revised_prompt) imageText(figure, item.revised_prompt, 'image-label');
        gallery.appendChild(figure);
    }
    if (!rendered) imageText(div, 'The response contained no displayable images.', 'image-error');
}

async function restoreImageResponse(message) {
    const div = imageMessageElement('image-response');
    imageText(div, 'Loading saved image…', 'image-label');
    try {
        const result = message.imageId ? await imageStore('get', message.imageId) : message.result;
        if (!result) throw new Error('Image is not saved in this browser. Download originals to keep them.');
        renderImageResponse(div, result, message.statusCode, message.duration);
    } catch (err) {
        div.innerHTML = '';
        imageText(div, err.message, 'image-error');
    }
}

async function saveImageResponse(session, div, result, statusCode, duration) {
    // Render first so quota/private-mode failures cannot lose the result.
    renderImageResponse(div, result, statusCode, duration);
    const message = {role: 'image-response', statusCode, duration};
    if (Array.isArray(result?.data) && statusCode < 400) {
        message.imageId = crypto.randomUUID();
        try {
            await imageStore('put', message.imageId, result);
        } catch (err) {
            delete message.imageId;
            message.result = {error: {message: 'Image was not saved in this browser. Download the original before leaving.'}};
            imageText(div, message.result.error.message, 'image-error');
        }
    } else {
        message.result = result;
    }
    session.messages.push(message);
    saveSessions();
    // Switching away and back rebuilt the pane while this request was pending.
    // Reuse the live result so a storage failure cannot hide its only preview.
    if (activeSessionId === session.id) {
        const container = document.getElementById('messages-container');
        if (!container.contains(div)) container.appendChild(div);
    }
}

async function sendImageMessage() {
    const input = document.getElementById('chat-input');
    const prompt = input.value.trim();
    if (!prompt || isStreaming) return;
    const target = selectedModelTarget();
    const operation = document.getElementById('image-operation').value || 'generations';
    if (!targetSupportsImageOperation(target, operation)) {
        appendMessageEl('error', 'Choose an image model supporting ' + operation + '.');
        return;
    }
    const n = Number(document.getElementById('image-count').value || 1);
    if (!Number.isInteger(n) || n < 1 || n > 4) {
        appendMessageEl('error', 'Image count must be between 1 and 4.');
        return;
    }
    if (operation === 'edits' && !imageReferenceFiles.length) {
        appendMessageEl('error', 'Choose at least one reference image to edit.');
        return;
    }
    if (operation === 'edits' && imageReferenceFiles.reduce((sum, f) => sum + f.size, 0) > 49 * 1024 * 1024) {
        appendMessageEl('error', 'Reference images must total less than 49 MiB.');
        return;
    }
    const payload = {
        model: target.id, prompt, n,
        size: document.getElementById('image-size').value || 'auto',
        quality: document.getElementById('image-quality').value || 'low',
        output_format: document.getElementById('image-format').value || 'png',
    };
    const options = {method: 'POST'};
    if (operation === 'edits') {
        const form = new FormData();
        for (const [key, value] of Object.entries(payload)) form.append(key, String(value));
        for (const file of imageReferenceFiles) form.append('image[]', file, file.name);
        options.body = form; // Browser supplies Content-Type and boundary.
    } else {
        options.headers = {'Content-Type': 'application/json'};
        options.body = JSON.stringify(payload);
    }
    let session = getActiveSession();
    if (!session) { newSession(); session = getActiveSession(); }
    session.model = document.getElementById('model-select').value;
    const endpoint = endpointForModel(null, operation);
    const content = `${prompt}\n${payload.size} · ${payload.quality} · ${n} ${payload.output_format}` +
        (operation === 'edits' ? '\nReferences: ' + imageReferenceFiles.map(f => f.name).join(', ') : '');
    session.messages.push({role: 'image-request', content, endpoint});
    if (session.messages.length === 1) {
        session.title = '[Image] ' + prompt.substring(0, 50);
        renderSessionList();
    }
    saveSessions();
    appendImageRequest(content, endpoint);
    const div = imageMessageElement('image-response');
    imageText(div, 'Generating image… This may take a while.', 'image-label');
    isStreaming = true;
    document.getElementById('send-btn').disabled = true;
    abortController = new AbortController();
    options.signal = abortController.signal;
    const start = performance.now();
    scrollToBottom();
    try {
        const response = await fetch(endpoint, options);
        const text = await response.text();
        let result;
        try { result = JSON.parse(text); } catch { result = {error: {message: text || 'Empty upstream response'}}; }
        await saveImageResponse(session, div, result, response.status, (performance.now() - start) / 1000);
    } catch (err) {
        imageText(div, err.message + ' No retry was attempted; check usage before submitting again.', 'image-error');
    } finally {
        isStreaming = false;
        abortController = null;
        document.getElementById('send-btn').disabled = false;
    }
    scrollToBottom();
}
