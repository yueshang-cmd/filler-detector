/* 会议/课程废话检测器 — M5 前端（Vue 3，无构建链）
 * API 契约以 app/main.py、app/jobs.py、app/settings.py、app/engines.py 为准：
 *  - POST /api/jobs            multipart(file, engine, model) -> {job_id, status}
 *  - GET  /api/jobs/{id}       -> {job_id, filename, engine, model, status, progress, error,
 *                                created_at, result, engine_fallback:{from,to,reason}, warnings[]}
 *  - GET  /api/jobs/{id}/result-> 直接返回 result 对象（兼容 {job, result} 包装），
 *                                含 engine_used / engine_fallback / warnings / keypoint_mode(offline|llm)
 *  - GET  /api/jobs            -> 上述公开视图列表（含 result）
 *  - GET  /api/audio/{id}      -> wav 流（支持 Range）
 *  - GET/PUT /api/settings     -> {default_engine, default_model, language(zh|auto), watchdog_seconds,
 *                                llm:{base_url,key,model,enabled}, filler_words:{pure[],contextual[]}}
 *                                注意：GET 返回明文 key（后端未做掩码）
 *  - GET  /api/engines         -> {cpu:{available,detail}, cuda:{available,detail}, rocm:{available,detail}}
 *                                （接口异常/404 时容错为仅 cpu 可用）
 */
/* global Vue */
'use strict';

var ACCEPT_EXTS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.mkv', '.mov', '.webm'];
var RING_R = 52;
var RING_C = 2 * Math.PI * RING_R;
var ACTIVE_STATUSES = ['queued', 'extracting', 'transcribing', 'analyzing'];

/* 兼容两种返回形态：直接 result 对象 / {job, result} 包装 */
function unwrapResult(payload) {
  if (payload && typeof payload === 'object' &&
      payload.result && typeof payload.result === 'object' && payload.result.segments) {
    return payload.result;
  }
  return payload;
}

var APP_TEMPLATE = `
<div class="page">
  <header class="hero">
    <div class="hero-text">
      <h1>会议 / 课程废话检测器</h1>
      <p class="sub">本地转写 · 废话浓度分析 · 重点提炼 —— 录音不出本机</p>
    </div>
    <button class="gear-btn" @click="openSettings" title="设置" aria-label="打开设置">
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
        <path fill="currentColor" d="M19.14 12.94a7.5 7.5 0 0 0 .05-.94 7.5 7.5 0 0 0-.05-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7 7 0 0 0-1.62-.94l-.36-2.54a.5.5 0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54a7 7 0 0 0-1.62.94l-2.39-.96a.5.5 0 0 0-.6.22L2.74 8.84a.5.5 0 0 0 .12.64l2.03 1.58a7.5 7.5 0 0 0 0 1.88l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32a.5.5 0 0 0 .6.22l2.39-.96a7 7 0 0 0 1.62.94l.36 2.54a.5.5 0 0 0 .5.42h3.84a.5.5 0 0 0 .5-.42l.36-2.54a7 7 0 0 0 1.62-.94l2.39.96a.5.5 0 0 0 .6-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58ZM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Z"/>
      </svg>
    </button>
  </header>

  <section class="card">
    <h2>1 · 上传音视频</h2>
    <div class="dropzone" :class="{ dragging: dragging, disabled: uploading }"
         @click="pickFile"
         @dragover.prevent="dragging = true"
         @dragleave.prevent="dragging = false"
         @drop.prevent="onDrop">
      <input ref="fileInput" type="file" class="hidden-input" :accept="acceptExts" @change="onFilePicked">
      <p v-if="!file" class="dz-main">拖拽文件到此处，或点击选择</p>
      <p v-else class="dz-main file-name">{{ file.name }} <span class="muted">（{{ fmtSize(file.size) }}）</span></p>
      <p class="muted small">支持 mp3 / wav / m4a / flac / ogg / mp4 / mkv / mov / webm</p>
    </div>
    <div class="controls">
      <label>引擎
        <select v-model="engine" :disabled="uploading">
          <option v-for="opt in engineOptions" :key="opt.value" :value="opt.value"
                  :disabled="!engines[opt.value]">{{ opt.label }}{{ engines[opt.value] ? '' : '（不可用：' + engineReason(opt.value) + '）' }}</option>
        </select>
      </label>
      <label>模型
        <select v-model="model" :disabled="uploading">
          <option v-for="opt in modelOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </label>
      <button class="primary" :disabled="!file || uploading || isActive" @click="startJob">
        {{ uploading ? '上传中 ' + uploadPercent + '%' : '开始分析' }}
      </button>
    </div>
    <p v-if="engineProbeNote" class="muted small">{{ engineProbeNote }}</p>
    <p v-if="submitError" class="error-text">{{ submitError }}</p>
  </section>

  <section class="card" v-if="jobStatus && (isActive || isFailed)">
    <h2>2 · 处理进度</h2>
    <p class="stage">{{ stageText }}</p>
    <div class="bar"><div class="bar-fill" :style="{ width: (jobStatus.progress || 0) + '%' }"></div></div>
    <div v-if="isFailed" class="error-box">
      <p><strong>任务失败：</strong>{{ jobStatus.error || '未知错误' }}</p>
      <p v-if="showFallbackHint" class="muted">检测到 GPU 相关错误：请改用 CPU 引擎重新提交（将回退 CPU 处理）。</p>
    </div>
  </section>

  <section class="card" v-if="result">
    <div class="result-head">
      <h2>3 · 分析结果</h2>
      <div class="result-tools">
        <select v-model="historyId" @change="onHistoryChange" title="切换历史任务">
          <option value="" disabled>历史任务…</option>
          <option v-for="j in doneJobs" :key="j.job_id" :value="j.job_id">{{ historyLabel(j) }}</option>
        </select>
        <button @click="exportMarkdown">导出 Markdown</button>
        <button @click="exportJSON">导出 JSON</button>
      </div>
    </div>
    <p class="muted small" v-if="resultJob">
      {{ resultJob.filename }} · 引擎 {{ resultJob.engine }} · 模型 {{ resultJob.model }} · {{ fmtDate(resultJob.created_at) }}
    </p>

    <div v-if="resultNotices.length" class="warn-bar" role="alert">
      <p v-for="(n, i) in resultNotices" :key="i" class="warn-line">{{ n }}</p>
    </div>

    <div class="gauges" v-if="analysis">
      <div class="gauge">
        <svg viewBox="0 0 120 120" class="ring">
          <circle cx="60" cy="60" r="52" class="ring-bg"></circle>
          <circle cx="60" cy="60" r="52" class="ring-fg ring-count"
                  :stroke-dasharray="ringDash(analysis.filler_count_ratio)"
                  transform="rotate(-90 60 60)"></circle>
        </svg>
        <div class="gauge-text">
          <span class="big">{{ pct(analysis.filler_count_ratio) }}</span>
          <span class="muted small">次数占比</span>
        </div>
      </div>
      <div class="gauge">
        <svg viewBox="0 0 120 120" class="ring">
          <circle cx="60" cy="60" r="52" class="ring-bg"></circle>
          <circle cx="60" cy="60" r="52" class="ring-fg ring-duration"
                  :stroke-dasharray="ringDash(analysis.filler_duration_ratio)"
                  transform="rotate(-90 60 60)"></circle>
        </svg>
        <div class="gauge-text">
          <span class="big">{{ pct(analysis.filler_duration_ratio) }}</span>
          <span class="muted small">时长占比</span>
        </div>
      </div>
      <div class="stats">
        <p>总词数 <strong>{{ analysis.total_tokens }}</strong></p>
        <p>废话次数 <strong>{{ analysis.filler_count }}</strong></p>
        <p>语音时长 <strong>{{ fmtTime(analysis.total_speech_duration) }}</strong>（{{ analysis.total_speech_duration }} 秒）</p>
        <p>音频时长 <strong>{{ fmtTime(result.duration) }}</strong> · 语言 {{ result.language }}</p>
      </div>
    </div>
    <p v-else class="empty-hint">该任务没有废话分析数据（早期版本结果），仅可查看转写与播放音频。</p>

    <audio ref="audio" controls preload="metadata" :src="audioSrc" class="player"></audio>

    <h3>废话 Top 10</h3>
    <table v-if="top10.length" class="top10">
      <thead><tr><th>词</th><th>次数</th><th>占用时长</th><th>例句跳转</th></tr></thead>
      <tbody>
        <tr v-for="item in top10" :key="item.word">
          <td class="filler-word">{{ item.word }}</td>
          <td>{{ item.count }}</td>
          <td>{{ item.duration }} 秒</td>
          <td>
            <button v-for="(ex, i) in item.examples" :key="i" class="ts-btn"
                    @click="seek(ex.start)" :title="ex.sentence">{{ fmtTime(ex.start) }}</button>
            <span v-if="!item.examples || !item.examples.length" class="muted">—</span>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-else class="empty-hint">{{ analysis ? '未检测到废话，表达很干净。' : '无分析数据。' }}</p>

    <h3>有用重点 <span class="badge" :class="keypointMode === 'llm' ? 'badge-llm' : 'badge-offline'">{{ keypointModeLabel }}</span></h3>
    <ol v-if="keypoints.length" class="keypoints">
      <li v-for="(kp, i) in keypoints" :key="i">
        <span class="stars" :title="'重要性 ' + kp.importance + ' / 5'">{{ stars(kp.importance) }}</span>
        <span class="kp-text">{{ kp.point }}</span>
        <button class="ts-btn" @click="seek(kp.start)">{{ fmtTime(kp.start) }}</button>
      </li>
    </ol>
    <p v-else class="empty-hint">未提炼出重点（音频过短或有效内容不足）。</p>

    <div class="transcript-head">
      <h3>完整转写</h3>
      <button @click="showTranscript = !showTranscript">{{ showTranscript ? '收起' : '展开' }}</button>
    </div>
    <div v-if="showTranscript" class="transcript">
      <p v-if="!transcriptSegments.length" class="empty-hint">转写为空。</p>
      <p v-for="(seg, i) in transcriptSegments" :key="i" class="seg">
        <button class="ts-btn" @click="seek(seg.start)">[{{ fmtTime(seg.start) }}]</button>
        <span v-for="(tok, j) in seg.tokens" :key="j" class="tok" :class="{ filler: tok.filler }"
              @click="seek(tok.start)"
              :title="tok.filler ? '废话词，点击跳转' : '点击跳转'">{{ tok.text }}</span>
      </p>
      <p class="muted small">红色高亮为判定出的废话词；点击时间戳或文字可跳转到对应音频位置。</p>
    </div>
  </section>

  <section class="card" v-if="!result && !isActive && !isFailed">
    <p class="empty-hint">上传文件开始分析，或从下方选择一个已完成的历史任务查看结果。</p>
    <div class="result-tools" v-if="doneJobs.length">
      <select v-model="historyId" @change="onHistoryChange">
        <option value="" disabled>查看历史任务…</option>
        <option v-for="j in doneJobs" :key="j.job_id" :value="j.job_id">{{ historyLabel(j) }}</option>
      </select>
    </div>
    <p v-else class="muted small">暂无历史任务。</p>
  </section>

  <footer class="muted small">filler-detector · 全程本地运行 · M5</footer>

  <div v-if="settingsOpen" class="modal-overlay" @click.self="closeSettings">
    <div class="modal" role="dialog" aria-modal="true" aria-label="设置">
      <div class="modal-head">
        <h2>设置</h2>
        <button class="modal-close" @click="closeSettings" title="关闭" aria-label="关闭">×</button>
      </div>

      <div class="modal-body">
        <p v-if="settingsLoading" class="muted">加载设置中…</p>
        <template v-else-if="settingsDraft">
          <section class="set-block">
            <h3>废话语料</h3>
            <p class="muted small">纯语气词无条件计为废话；语境词需满足启发式规则才计入。</p>

            <div class="word-group">
              <label class="word-label">纯语气词</label>
              <div class="tags">
                <span v-for="(w, i) in settingsDraft.filler_words.pure" :key="'p' + i" class="tag">
                  {{ w }}<button class="tag-x" @click="removeWord('pure', i)" :aria-label="'删除 ' + w">×</button>
                </span>
                <span v-if="!settingsDraft.filler_words.pure.length" class="muted small">（空）</span>
              </div>
              <div class="tag-add">
                <input v-model="newPure" type="text" placeholder="输入词后回车或点添加" @keydown.enter.prevent="addPure">
                <button @click="addPure">添加</button>
              </div>
            </div>

            <div class="word-group">
              <label class="word-label">语境词</label>
              <div class="tags">
                <span v-for="(w, i) in settingsDraft.filler_words.contextual" :key="'c' + i" class="tag">
                  {{ w }}<button class="tag-x" @click="removeWord('contextual', i)" :aria-label="'删除 ' + w">×</button>
                </span>
                <span v-if="!settingsDraft.filler_words.contextual.length" class="muted small">（空）</span>
              </div>
              <div class="tag-add">
                <input v-model="newContextual" type="text" placeholder="输入词后回车或点添加" @keydown.enter.prevent="addContextual">
                <button @click="addContextual">添加</button>
              </div>
            </div>
          </section>

          <section class="set-block">
            <h3>重点提炼 LLM</h3>
            <p class="muted small">配置后重点提炼改用 LLM，失败自动回退离线。</p>
            <label class="switch">
              <input type="checkbox" v-model="settingsDraft.llm.enabled">
              <span>启用 LLM 提炼</span>
            </label>
            <div class="form-grid">
              <label>Base URL
                <input type="text" v-model="settingsDraft.llm.base_url" placeholder="https://api.example.com/v1">
              </label>
              <label>模型
                <input type="text" v-model="settingsDraft.llm.model" placeholder="gpt-4o-mini 等">
              </label>
              <label>API Key
                <input type="password" v-model="settingsDraft.llm.key" placeholder="仅保存在本机 settings.json" autocomplete="off">
              </label>
            </div>
          </section>

          <section class="set-block">
            <h3>默认转写参数</h3>
            <div class="form-grid">
              <label>默认引擎
                <select v-model="settingsDraft.default_engine">
                  <option v-for="opt in engineOptions" :key="opt.value" :value="opt.value"
                          :disabled="!engines[opt.value]">{{ opt.label }}{{ engines[opt.value] ? '' : '（不可用：' + engineReason(opt.value) + '）' }}</option>
                </select>
              </label>
              <label>默认模型
                <select v-model="settingsDraft.default_model">
                  <option v-for="opt in modelOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
                </select>
              </label>
              <label>语言
                <select v-model="settingsDraft.language">
                  <option value="zh">中文（zh）</option>
                  <option value="auto">自动检测（auto）</option>
                </select>
              </label>
              <label>看门狗超时（秒）
                <input type="number" min="1" step="1" v-model.number="settingsDraft.watchdog_seconds">
              </label>
            </div>
            <p class="muted small">GPU 引擎无进度超过该秒数将中止并回退 CPU。</p>
          </section>
        </template>
      </div>

      <div class="modal-foot">
        <span v-if="settingsMsg" class="ok-text">{{ settingsMsg }}</span>
        <span v-if="settingsErr" class="error-text">{{ settingsErr }}</span>
        <span class="spacer"></span>
        <button @click="closeSettings">取消</button>
        <button class="primary" :disabled="settingsSaving || settingsLoading || !settingsDraft" @click="saveSettings">
          {{ settingsSaving ? '保存中…' : '保存' }}
        </button>
      </div>
    </div>
  </div>
</div>
`;

Vue.createApp({
  template: APP_TEMPLATE,

  data: function () {
    return {
      file: null,
      dragging: false,
      acceptExts: ACCEPT_EXTS.join(','),
      engine: 'cpu',
      model: 'small',
      engines: { cpu: true, cuda: false, rocm: false },
      engineDetails: { cpu: '', cuda: '', rocm: '' },
      engineProbeNote: '引擎探测中…',
      engineOptions: [
        { value: 'cpu', label: 'CPU（默认）' },
        { value: 'cuda', label: 'CUDA GPU' },
        { value: 'rocm', label: 'ROCm GPU' }
      ],
      modelOptions: [
        { value: 'small', label: 'small（快，默认）' },
        { value: 'medium', label: 'medium（更准，较慢）' },
        { value: 'large-v3', label: 'large-v3（最准，最慢）' }
      ],
      uploading: false,
      uploadPercent: 0,
      submitError: '',
      currentJobId: null,
      jobStatus: null,
      pollTimer: null,
      result: null,
      resultJob: null,
      history: [],
      historyId: '',
      showTranscript: false,
      settingsOpen: false,
      settingsLoading: false,
      settingsSaving: false,
      settingsMsg: '',
      settingsErr: '',
      settingsCache: null,
      settingsDraft: null,
      newPure: '',
      newContextual: ''
    };
  },

  computed: {
    isActive: function () {
      return !!this.jobStatus && ACTIVE_STATUSES.indexOf(this.jobStatus.status) !== -1;
    },
    isFailed: function () {
      return !!this.jobStatus && this.jobStatus.status === 'failed';
    },
    stageText: function () {
      if (!this.jobStatus) return '';
      var p = this.jobStatus.progress || 0;
      switch (this.jobStatus.status) {
        case 'queued': return '排队中…';
        case 'extracting': return '抽取音频中…';
        case 'transcribing': return '转写中 ' + p + '%';
        case 'analyzing': return '分析中（废话判定与重点提炼）…';
        case 'done': return '已完成';
        case 'failed': return '失败';
        default: return this.jobStatus.status;
      }
    },
    showFallbackHint: function () {
      if (!this.isFailed) return false;
      var err = this.jobStatus.error || '';
      var ws = (this.jobStatus.warnings || []).join(' ');
      return /gpu|cuda|rocm|回退|fallback/i.test(err + ' ' + ws);
    },
    analysis: function () {
      return (this.result && this.result.analysis) || null;
    },
    top10: function () {
      return (this.analysis && this.analysis.top10) || [];
    },
    keypoints: function () {
      var kp = (this.result && this.result.keypoints) || [];
      return kp.slice().sort(function (a, b) { return (b.importance || 0) - (a.importance || 0); });
    },
    keypointMode: function () {
      return (this.result && this.result.keypoint_mode) || 'offline';
    },
    keypointModeLabel: function () {
      return this.keypointMode === 'llm' ? 'LLM 提炼' : '离线算法';
    },
    resultWarnings: function () {
      var w = (this.result && this.result.warnings) ||
              (this.resultJob && this.resultJob.warnings) || [];
      return Array.isArray(w) ? w : [];
    },
    engineFallback: function () {
      return (this.result && this.result.engine_fallback) ||
             (this.resultJob && this.resultJob.engine_fallback) || null;
    },
    resultNotices: function () {
      var warnings = this.resultWarnings;
      if (warnings.length) return warnings.slice();
      var fb = this.engineFallback;
      if (fb && fb.from && fb.to) {
        return [fb.from + ' 引擎失败已回退 ' + fb.to + (fb.reason ? '：' + fb.reason : '')];
      }
      return [];
    },
    audioSrc: function () {
      return this.currentJobId ? '/api/audio/' + this.currentJobId : '';
    },
    doneJobs: function () {
      return this.history.filter(function (j) { return j.status === 'done'; });
    },
    transcriptSegments: function () {
      var segs = (this.result && this.result.segments) || [];
      var fillers = (this.analysis && this.analysis.fillers) || [];
      var self = this;
      return segs.map(function (seg) {
        return { start: seg.start, tokens: self.buildTokens(seg, fillers) };
      });
    }
  },

  methods: {
    /* ---------- 格式化 ---------- */
    fmtTime: function (sec) {
      if (sec == null || isNaN(sec)) return '--:--';
      var s = Math.max(0, Math.floor(Number(sec)));
      var m = Math.floor(s / 60);
      var r = s % 60;
      return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0');
    },
    fmtSize: function (bytes) {
      if (bytes == null) return '';
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
      if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
      return (bytes / 1073741824).toFixed(2) + ' GB';
    },
    fmtDate: function (ts) {
      if (!ts) return '';
      var d = new Date(ts * 1000);
      if (isNaN(d.getTime())) return '';
      function p2(n) { return String(n).padStart(2, '0'); }
      return d.getFullYear() + '-' + p2(d.getMonth() + 1) + '-' + p2(d.getDate()) +
        ' ' + p2(d.getHours()) + ':' + p2(d.getMinutes());
    },
    pct: function (ratio) {
      return ((Number(ratio) || 0) * 100).toFixed(1) + '%';
    },
    ringDash: function (ratio) {
      var v = Math.max(0, Math.min(1, Number(ratio) || 0));
      return (RING_C * v).toFixed(2) + ' ' + RING_C.toFixed(2);
    },
    stars: function (n) {
      var k = Math.max(0, Math.min(5, Math.round(Number(n) || 0)));
      return '★'.repeat(k) + '☆'.repeat(5 - k);
    },
    historyLabel: function (j) {
      return (j.filename || j.job_id) + ' · ' + this.fmtDate(j.created_at) + ' · ' + j.job_id;
    },

    /* ---------- 文件选择 ---------- */
    pickFile: function () {
      if (!this.uploading && this.$refs.fileInput) this.$refs.fileInput.click();
    },
    onFilePicked: function (e) {
      var f = e.target.files && e.target.files[0];
      if (f) this.setFile(f);
      e.target.value = '';
    },
    onDrop: function (e) {
      this.dragging = false;
      if (this.uploading) return;
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) this.setFile(f);
    },
    setFile: function (f) {
      var lower = (f.name || '').toLowerCase();
      var ok = ACCEPT_EXTS.some(function (ext) { return lower.endsWith(ext); });
      if (!ok) {
        this.submitError = '不支持的文件类型：' + f.name;
        this.file = null;
        return;
      }
      this.submitError = '';
      this.file = f;
    },

    /* ---------- 设置 / 引擎探测 / 历史 ---------- */
    loadSettings: function () {
      var self = this;
      return fetch('/api/settings').then(function (r) {
        if (!r.ok) return null;
        return r.json();
      }).then(function (s) {
        if (!s) return;
        if (s.default_model) self.model = s.default_model;
        if (s.default_engine) self.engine = s.default_engine;
        self.applyEngineAvailability();
      }).catch(function () { /* 设置不可用时保留默认值 */ });
    },
    probeEngines: function () {
      var self = this;
      return fetch('/api/engines').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (data) {
        var parsed = self.parseEngines(data);
        self.engines = parsed.avail;
        self.engineDetails = parsed.details;
        self.engineProbeNote = '';
        self.applyEngineAvailability();
      }).catch(function () {
        /* /api/engines 不可用（如 404）：容错为仅 CPU 可用 */
        self.engines = { cpu: true, cuda: false, rocm: false };
        self.engineDetails = { cpu: '', cuda: '', rocm: '' };
        self.engineProbeNote = '引擎探测接口不可用，当前仅 CPU 可用，CUDA / ROCm 已置灰。';
        self.applyEngineAvailability();
      });
    },
    parseEngines: function (data) {
      var avail = { cpu: true, cuda: false, rocm: false };
      var details = { cpu: '', cuda: '', rocm: '' };
      var src = (data && data.engines && typeof data.engines === 'object') ? data.engines : data;
      if (src && typeof src === 'object') {
        ['cpu', 'cuda', 'rocm'].forEach(function (k) {
          var v = src[k];
          if (typeof v === 'boolean') {
            avail[k] = v;
          } else if (v && typeof v === 'object') {
            if (typeof v.available === 'boolean') avail[k] = v.available;
            if (typeof v.detail === 'string') details[k] = v.detail;
          }
        });
      }
      avail.cpu = true; /* cpu 恒可用 */
      return { avail: avail, details: details };
    },
    engineReason: function (value) {
      var d = (this.engineDetails && this.engineDetails[value]) || '';
      d = String(d).replace(/\s+/g, ' ').trim();
      if (!d) return '未探测到可用设备';
      return d.length > 40 ? d.slice(0, 40) + '…' : d;
    },
    applyEngineAvailability: function () {
      if (!this.engines[this.engine]) this.engine = 'cpu';
    },
    loadHistory: function () {
      var self = this;
      return fetch('/api/jobs').then(function (r) {
        if (!r.ok) return;
        return r.json();
      }).then(function (list) {
        if (Array.isArray(list)) self.history = list;
      }).catch(function () { /* 历史不可用不阻塞主流程 */ });
    },

    /* ---------- 设置面板 ---------- */
    cloneSettings: function (s) {
      s = s || {};
      var fw = s.filler_words || {};
      var llm = s.llm || {};
      return {
        default_engine: s.default_engine || 'cpu',
        default_model: s.default_model || 'small',
        language: s.language === 'auto' ? 'auto' : 'zh',
        watchdog_seconds: (s.watchdog_seconds != null ? s.watchdog_seconds : 600),
        llm: {
          base_url: typeof llm.base_url === 'string' ? llm.base_url : '',
          key: typeof llm.key === 'string' ? llm.key : '',
          model: typeof llm.model === 'string' ? llm.model : '',
          enabled: !!llm.enabled
        },
        filler_words: {
          pure: Array.isArray(fw.pure) ? fw.pure.slice() : [],
          contextual: Array.isArray(fw.contextual) ? fw.contextual.slice() : []
        }
      };
    },
    openSettings: function () {
      var self = this;
      this.settingsOpen = true;
      this.settingsMsg = '';
      this.settingsErr = '';
      this.settingsLoading = true;
      this.settingsDraft = null;
      this.newPure = '';
      this.newContextual = '';
      fetch('/api/settings').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (s) {
        self.settingsCache = s;
        self.settingsDraft = self.cloneSettings(s);
        self.settingsLoading = false;
      }).catch(function (e) {
        self.settingsLoading = false;
        self.settingsErr = '读取设置失败：' + e.message;
        self.settingsDraft = self.cloneSettings(self.settingsCache || {});
      });
    },
    closeSettings: function () {
      this.settingsOpen = false;
      this.settingsDraft = null;
      this.settingsMsg = '';
      this.settingsErr = '';
    },
    addPure: function () {
      var w = (this.newPure || '').trim();
      if (!w || !this.settingsDraft) return;
      if (this.settingsDraft.filler_words.pure.indexOf(w) === -1) {
        this.settingsDraft.filler_words.pure.push(w);
      }
      this.newPure = '';
    },
    addContextual: function () {
      var w = (this.newContextual || '').trim();
      if (!w || !this.settingsDraft) return;
      if (this.settingsDraft.filler_words.contextual.indexOf(w) === -1) {
        this.settingsDraft.filler_words.contextual.push(w);
      }
      this.newContextual = '';
    },
    removeWord: function (kind, index) {
      if (!this.settingsDraft || !this.settingsDraft.filler_words[kind]) return;
      this.settingsDraft.filler_words[kind].splice(index, 1);
    },
    saveSettings: function () {
      var self = this;
      if (!this.settingsDraft || this.settingsSaving) return;
      this.settingsMsg = '';
      this.settingsErr = '';
      var d = this.settingsDraft;
      var wd = Number(d.watchdog_seconds);
      if (!isFinite(wd) || wd <= 0) {
        this.settingsErr = '看门狗秒数必须为大于 0 的数字';
        return;
      }
      if (d.llm.enabled && (!d.llm.base_url.trim() || !d.llm.model.trim())) {
        this.settingsErr = '启用 LLM 时 Base URL 与模型不能为空';
        return;
      }
      var payload = {
        default_engine: d.default_engine,
        default_model: d.default_model,
        language: d.language,
        watchdog_seconds: wd,
        llm: {
          base_url: d.llm.base_url,
          key: d.llm.key,
          model: d.llm.model,
          enabled: !!d.llm.enabled
        },
        filler_words: {
          pure: d.filler_words.pure.slice(),
          contextual: d.filler_words.contextual.slice()
        }
      };
      this.settingsSaving = true;
      fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          return { ok: r.ok, status: r.status, body: body };
        });
      }).then(function (res) {
        self.settingsSaving = false;
        if (!res.ok) {
          var msg = (res.body && res.body.detail) ? res.body.detail : ('HTTP ' + res.status);
          self.settingsErr = '保存失败：' + msg;
          return;
        }
        self.settingsCache = res.body;
        self.settingsDraft = self.cloneSettings(res.body);
        self.settingsMsg = '设置已保存';
      }).catch(function (e) {
        self.settingsSaving = false;
        self.settingsErr = '保存失败：' + e.message;
      });
    },

    /* ---------- 上传与轮询 ---------- */
    startJob: function () {
      if (!this.file || this.uploading) return;
      var self = this;
      this.submitError = '';
      this.uploading = true;
      this.uploadPercent = 0;
      var fd = new FormData();
      fd.append('file', this.file);
      fd.append('engine', this.engine);
      fd.append('model', this.model);
      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/jobs');
      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) self.uploadPercent = Math.round(e.loaded / e.total * 100);
      };
      xhr.onload = function () {
        self.uploading = false;
        if (xhr.status >= 200 && xhr.status < 300) {
          var data = {};
          try { data = JSON.parse(xhr.responseText); } catch (e) { /* 忽略 */ }
          if (!data.job_id) {
            self.submitError = '服务器未返回 job_id';
            return;
          }
          self.currentJobId = data.job_id;
          self.jobStatus = {
            job_id: data.job_id,
            status: data.status || 'queued',
            progress: 0,
            error: null,
            filename: self.file ? self.file.name : '',
            engine: self.engine,
            model: self.model
          };
          self.result = null;
          self.resultJob = null;
          self.showTranscript = false;
          self.startPolling();
        } else {
          var msg = '上传失败（HTTP ' + xhr.status + '）';
          try {
            var d = JSON.parse(xhr.responseText);
            if (d.detail) msg += '：' + d.detail;
          } catch (e) { /* 忽略 */ }
          self.submitError = msg;
        }
      };
      xhr.onerror = function () {
        self.uploading = false;
        self.submitError = '网络错误，上传失败';
      };
      xhr.onabort = function () {
        self.uploading = false;
        self.submitError = '上传已中止';
      };
      xhr.send(fd);
    },
    startPolling: function () {
      this.stopPolling();
      var self = this;
      this.pollTimer = setInterval(function () { self.pollOnce(); }, 2000);
      this.pollOnce();
    },
    stopPolling: function () {
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    },
    pollOnce: function () {
      if (!this.currentJobId) return;
      var self = this;
      var id = this.currentJobId;
      return fetch('/api/jobs/' + id).then(function (r) {
        if (r.status === 404) {
          self.stopPolling();
          self.submitError = '任务不存在（服务重启后内存任务可能丢失），请重新上传';
          return;
        }
        if (!r.ok) return;
        return r.json();
      }).then(function (view) {
        if (!view || self.currentJobId !== id) return;
        self.jobStatus = view;
        if (view.status === 'done') {
          self.stopPolling();
          self.adoptResult(id, view, view.result);
          self.loadHistory();
        } else if (view.status === 'failed') {
          self.stopPolling();
        }
      }).catch(function () { /* 单次轮询失败忽略，下个周期重试 */ });
    },

    /* ---------- 结果加载 / 历史切换 ---------- */
    adoptResult: function (jobId, meta, embedded) {
      var self = this;
      this.currentJobId = jobId;
      this.historyId = jobId;
      this.resultJob = meta || null;
      this.showTranscript = false;
      var payload = embedded || null;
      var ready = (payload && payload.segments) ? Promise.resolve(payload) :
        fetch('/api/jobs/' + jobId + '/result').then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        });
      return ready.then(function (data) {
        self.result = unwrapResult(data);
      }).catch(function (e) {
        self.submitError = '读取结果失败：' + e.message;
      });
    },
    onHistoryChange: function () {
      var id = this.historyId;
      if (!id) return;
      var job = this.history.find(function (j) { return j.job_id === id; }) || null;
      this.stopPolling();
      this.submitError = '';
      this.jobStatus = job;
      this.adoptResult(id, job, job && job.result);
    },

    /* ---------- 播放器跳转 ---------- */
    seek: function (t) {
      var audio = this.$refs.audio;
      if (!audio || t == null || isNaN(t)) return;
      try {
        audio.currentTime = Math.max(0, Number(t));
        var p = audio.play();
        if (p && p.catch) p.catch(function () { /* 自动播放被拦截时忽略 */ });
      } catch (e) { /* 音频尚未就绪时忽略 */ }
    },

    /* ---------- 转写高亮：fillers 与 words 按时间对齐 ---------- */
    buildTokens: function (seg, fillers) {
      var words = (seg && seg.words) || [];
      if (!words.length) {
        return [{ text: (seg && seg.text) || '', start: seg ? seg.start : 0, filler: false }];
      }
      return words.map(function (w) {
        var cleaned = String(w.word || '').replace(/[^\p{L}\p{N}]/gu, '').toLowerCase();
        var isFiller = false;
        if (cleaned) {
          for (var i = 0; i < fillers.length; i++) {
            var f = fillers[i];
            var fw = String(f.word || '').toLowerCase();
            var ws = Number(w.start) || 0;
            var fs = Number(f.start) || 0;
            var fe = Number(f.end) || 0;
            var timeNear = Math.abs(ws - fs) < 0.06 || (ws >= fs - 0.06 && ws < fe + 0.001);
            if (timeNear && (cleaned === fw || cleaned.indexOf(fw) !== -1 || fw.indexOf(cleaned) !== -1)) {
              isFiller = true;
              break;
            }
          }
        }
        return { text: w.word, start: w.start, filler: isFiller };
      });
    },

    /* ---------- 导出 ---------- */
    download: function (filename, text, mime) {
      var blob = new Blob([text], { type: mime });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    },
    exportJSON: function () {
      if (!this.result) return;
      this.download(
        'filler-report-' + (this.currentJobId || 'result') + '.json',
        JSON.stringify(this.result, null, 2),
        'application/json;charset=utf-8'
      );
    },
    exportMarkdown: function () {
      if (!this.result) return;
      var r = this.result;
      var a = this.analysis;
      var meta = this.resultJob || {};
      var self = this;
      var L = [];
      L.push('# 废话分析报告');
      L.push('');
      L.push('- 文件：' + (meta.filename || '未知'));
      L.push('- 任务 ID：' + (this.currentJobId || ''));
      L.push('- 引擎 / 模型：' + (meta.engine || '') + ' / ' + (meta.model || ''));
      L.push('- 生成时间：' + new Date().toLocaleString('zh-CN'));
      L.push('- 音频时长：' + this.fmtTime(r.duration) + '（' + r.duration + ' 秒）');
      L.push('- 语言：' + (r.language || ''));
      L.push('');
      if (a) {
        L.push('## 废话浓度');
        L.push('');
        L.push('| 口径 | 占比 | 明细 |');
        L.push('| --- | --- | --- |');
        L.push('| 次数占比 | ' + this.pct(a.filler_count_ratio) + ' | ' + a.filler_count + ' 次 / ' + a.total_tokens + ' 词 |');
        L.push('| 时长占比 | ' + this.pct(a.filler_duration_ratio) + ' | ' + a.filler_duration + ' 秒 / ' + a.total_speech_duration + ' 秒语音 |');
        L.push('');
        L.push('## 废话 Top 10');
        L.push('');
        if (this.top10.length) {
          L.push('| 词 | 次数 | 占用时长 | 例句时间戳 |');
          L.push('| --- | --- | --- | --- |');
          this.top10.forEach(function (item) {
            var ts = (item.examples || []).map(function (ex) { return self.fmtTime(ex.start); }).join('、');
            L.push('| ' + item.word + ' | ' + item.count + ' | ' + item.duration + ' 秒 | ' + (ts || '—') + ' |');
          });
        } else {
          L.push('未检测到废话。');
        }
        L.push('');
      } else {
        L.push('（该任务无废话分析数据）');
        L.push('');
      }
      L.push('## 有用重点（' + this.keypointModeLabel + '）');
      L.push('');
      if (this.keypoints.length) {
        this.keypoints.forEach(function (kp) {
          L.push('- ' + self.stars(kp.importance) + ' ' + kp.point + ' [' + self.fmtTime(kp.start) + ']');
        });
      } else {
        L.push('未提炼出重点。');
      }
      L.push('');
      L.push('## 完整转写');
      L.push('');
      var segs = r.segments || [];
      if (segs.length) {
        segs.forEach(function (seg) {
          L.push('[' + self.fmtTime(seg.start) + '] ' + seg.text);
          L.push('');
        });
      } else {
        L.push('转写为空。');
      }
      this.download(
        'filler-report-' + (this.currentJobId || 'result') + '.md',
        L.join('\n'),
        'text/markdown;charset=utf-8'
      );
    }
  },

  mounted: function () {
    this.loadSettings();
    this.probeEngines();
    this.loadHistory();
    var self = this;
    this._onKey = function (e) {
      if (e.key === 'Escape' && self.settingsOpen) self.closeSettings();
    };
    if (typeof document !== 'undefined' && document.addEventListener) {
      document.addEventListener('keydown', this._onKey);
    }
  },
  beforeUnmount: function () {
    this.stopPolling();
    if (typeof document !== 'undefined' && document.removeEventListener && this._onKey) {
      document.removeEventListener('keydown', this._onKey);
    }
  }
}).mount('#app');