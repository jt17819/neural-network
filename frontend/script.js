const canvas = document.querySelector("canvas")
const clearCanvas = document.querySelector(".clear-canvas")
const sendImg = document.querySelector(".send-img")
const advanced = document.querySelector("#advanced")
let ctx = canvas.getContext("2d")

let isDrawing = false
let brushWidth = Math.floor(canvas.offsetWidth / 12)

const TF_URL = "https://tensorflow-neural-network.onrender.com/predict"

const setCanvasBackground = () => {
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = "#000"
}

window.addEventListener("load", () => {
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight
    setCanvasBackground()
})

const startDrawing = () => {
    isDrawing = true
    ctx.beginPath()
    ctx.lineWidth = brushWidth
    ctx.fillStyle = "#000"
}

const stopDrawing = () => {
    isDrawing = false
}

const drawing = (e) => {
    if (!isDrawing) return
    ctx.lineTo(e.offsetX, e.offsetY)
    ctx.stroke()
}

const postFetch = async (payload) => {
    const options = {
        method: "POST",
        body: JSON.stringify({"payload": payload}),
        headers: new Headers({"X-Api-Key": "AxZaVHIQOY1E9rkq2tiNu3p8IbDAi7FN9K667KSB", "Content-Type": "application/x-www-form-urlencoded"})
    }
    const resp = await fetch("https://8g46y1f790.execute-api.eu-west-2.amazonaws.com/default/neural-network-demo", options)
    return resp.json()
}

const pyTorchPost = async (payload) => {
    const options = {
        method: "POST",
        body: JSON.stringify({"payload": payload}),
        headers: new Headers({"Content-Type": "application/json"})
    }
    const resp = await fetch("https://neural-network-7zl9.onrender.com/predict", options)
    return resp.json()
}

const tensorflowPost = async (payload) => {
    const options = {
        method: "POST",
        body: JSON.stringify({"payload": payload}),
        headers: new Headers({"Content-Type": "application/json"})
    }
    const resp = await fetch(TF_URL, options)
    return resp.json()
}

// Handle both 2D [[...]] and 1D [...] raw output formats
const getRaw = (resp) => Array.isArray(resp["raw"][0]) ? resp["raw"][0] : resp["raw"]

const lastResults = { "my-nn": null, "pytorch": null, "tensorflow": null }

const showModelLoading = (modelId) => {
    document.querySelector(`#result-${modelId} .model-output`).innerHTML = `<span class="loading">Loading...</span>`
}

const showModelResult = (modelId, resp, isError = false) => {
    const output = document.querySelector(`#result-${modelId} .model-output`)
    output.innerHTML = ""

    if (isError) {
        output.innerHTML = `<span class="error">Error</span>`
        return
    }

    lastResults[modelId] = resp

    if (advanced.checked) {
        const raw = getRaw(resp)
        const list = document.createElement("ul")
        for (let i = 0; i < 10; i++) {
            const item = document.createElement("li")
            item.textContent = `${i}: ${(raw[i] * 100).toFixed(2)}%`
            if (i == resp["ans"]) item.classList.add("answer")
            list.appendChild(item)
        }
        output.appendChild(list)
    } else {
        output.innerHTML = `<h3>${resp["ans"]}</h3>`
    }
}

advanced.addEventListener("change", () => {
    for (const [id, resp] of Object.entries(lastResults)) {
        if (resp !== null) showModelResult(id, resp)
    }
})

clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    setCanvasBackground()
    document.querySelectorAll(".model-output").forEach(el => el.innerHTML = "")
    lastResults["my-nn"] = null
    lastResults["pytorch"] = null
    lastResults["tensorflow"] = null
})

sendImg.addEventListener("click", async (e) => {
    e.target.disabled = true
    setTimeout(() => { e.target.disabled = false }, 5000)

    const payload = canvas.toDataURL("image/jpg")

    showModelLoading("my-nn")
    showModelLoading("pytorch")
    showModelLoading("tensorflow")

    const run = (id, promise) =>
        promise
            .then(resp => showModelResult(id, resp, false))
            .catch(() => showModelResult(id, null, true))

    run("my-nn", postFetch(payload))
    run("pytorch", pyTorchPost(payload))
    run("tensorflow", tensorflowPost(payload))
})

canvas.addEventListener("mousedown", startDrawing)
canvas.addEventListener("mouseup", stopDrawing)
canvas.addEventListener("mousemove", drawing)

canvas.addEventListener("touchstart", startDrawing)
canvas.addEventListener("touchend", stopDrawing)
canvas.addEventListener("touchmove", drawing)

