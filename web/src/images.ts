/** Longest side, in pixels, that images are scaled down to before upload. The embedders work at
 * about 224 px, so this keeps plenty of detail for them and for viewing while making uploads of
 * camera photos many times smaller. */
export const MAX_UPLOAD_SIDE = 1024;

/** Size in pixels at which the server is asked for example thumbnails (2x the 120 px tiles). */
export const THUMB_SIZE = 240;

/** ``file`` scaled down to fit in ``MAX_UPLOAD_SIDE`` and re-encoded as JPEG, or ``file`` itself
 * when it is already small enough or the browser can't decode it (the server then decides). */
export async function shrinkImage(file: File): Promise<File> {
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); // bakes in EXIF rotation
  } catch {
    return file;
  }
  try {
    const scale = MAX_UPLOAD_SIDE / Math.max(bmp.width, bmp.height);
    if (scale >= 1) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bmp.width * scale));
    canvas.height = Math.max(1, Math.round(bmp.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.fillStyle = "#fff"; // JPEG has no alpha: show transparent areas as white
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((res) => { canvas.toBlob(res, "image/jpeg", 0.9); });
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.[^.]*$/, "") + ".jpg", { type: "image/jpeg", lastModified: file.lastModified });
  } finally {
    bmp.close();
  }
}

/** ``shrinkImage`` over several files, a few at a time to bound memory. */
export async function shrinkImages(files: File[]): Promise<File[]> {
  const out: File[] = new Array<File>(files.length);
  let next = 0;
  const worker = async () => {
    while (next < files.length) {
      const i = next++;
      out[i] = await shrinkImage(files[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, files.length) }, worker));
  return out;
}
